#!/usr/bin/env python3
"""Decode a `DUMP KERNEL ARGS` hex dump from an HSA QUEUE ANALYSIS crash log.

Generic across kernel families — works for PyTorch aten / Triton / CK GEMM /
custom HIP kernels, not just PyTorch. Two modes:

1) ABI mode (default) — you know the kernel's argument layout (from its launch
   site source). Feed a JSON ABI spec. Supports flat scalars/pointers and the
   PyTorch TensorInfo compound. Default ABI = PyTorch `indexSelectSmallIndex`.

2) --raw blind mode — no ABI needed. Scans every u64 word (potential pointer)
   and the scalar region. With --gpucore, cross-references each pointer against
   the gpucore PT_LOAD segments to identify which buffer it points into.

Usage:
    python parse_kernel_args.py CRASH.log                      # ABI mode, default ABI
    python parse_kernel_args.py CRASH.log ABI.json             # ABI mode, custom
    python parse_kernel_args.py CRASH.log --raw                # blind, no ABI
    python parse_kernel_args.py CRASH.log --raw --gpucore gpucore.21292
    python parse_kernel_args.py CRASH.log ABI.json --gpucore gpucore.21292
    python parse_kernel_args.py CRASH.log --size 928           # override buffer size

Custom ABI JSON (fields in order; offsets absolute in the args buffer):
    {
      "name": "myKernel",
      "size": 928,                     # args buffer size (from the log)
      "tensorinfo_dtype": "uint",      # TensorInfo IndexType; sizes/strides always int32
      "tensorinfo_dim_cap": 25,        # max dims -> 216-byte TensorInfo
      "fields": [
        {"name": "a", "type": "u64",     "offset": 0},   # pointer / plain u64
        {"name": "n", "type": "i32",     "offset": 8},   # scalar
        {"name": "t", "type": "tensorinfo", "offset": 16} # PyTorch TensorInfo
      ]
    }
Scalar/pointer types: i8/i16/i32/i64/u8/u16/u32/u64/f32/f64.
TensorInfo layout (cap=25): data@0 (8B), sizes[cap]@8 (4B), strides@(8+4*cap)
(4B), dims@(8+4*cap+4*cap) (4B), pad -> 216 bytes.
"""
import argparse
import json
import re
import struct

DEFAULT_ABI = {
    "name": "indexSelectSmallIndex",
    "size": 928,
    "tensorinfo_dtype": "uint",
    "tensorinfo_dim_cap": 25,
    "fields": [
        {"name": "dst", "type": "tensorinfo", "offset": 0},
        {"name": "src", "type": "tensorinfo", "offset": 216},
        {"name": "indices", "type": "tensorinfo", "offset": 432},
        {"name": "dstSelectDim", "type": "i32", "offset": 648},
        {"name": "srcSelectDim", "type": "i32", "offset": 652},
        {"name": "innerSize", "type": "u32", "offset": 656},
        {"name": "srcSelectDimSize", "type": "i64", "offset": 664},
    ],
}

SCALAR_FORMATS = {
    "i8": "<b", "u8": "<B", "i16": "<h", "u16": "<H",
    "i32": "<i", "u32": "<I", "i64": "<q", "u64": "<Q",
    "f32": "<f", "f64": "<d",
}
SIZE_MARKER = re.compile(r"DUMP KERNEL ARGS:\s*size:\s*(\d+)")


def read_lines(path):
    """Split on \\n like grep (NOT splitlines: bare \\r progress-carriage
    returns are miscounted as extra lines and can corrupt hex extraction)."""
    raw = open(path, "rb").read().split(b"\n")
    return [ln.rstrip(b"\r").decode("utf-8", "replace") for ln in raw]


def find_size(lines):
    """Auto-detect the args buffer size from the first DUMP KERNEL ARGS line."""
    for l in lines:
        m = SIZE_MARKER.search(l)
        if m:
            return int(m.group(1))
    raise SystemExit("DUMP KERNEL ARGS size not found; pass --size N")


def find_arg_dump_start(lines, size):
    marker = f"DUMP KERNEL ARGS: size: {size}"
    for i, l in enumerate(lines):
        if marker in l:
            return i + 1
    raise SystemExit(f"arg dump marker not found: {marker!r}")


def extract_blob(lines, start, size):
    blob = b""
    for l in lines[start:]:
        if "DUMP KERNEL ARGS PTR INFO" in l:
            break
        hexs = re.findall(r"\b[0-9a-fA-F]{2}\b", l)
        if len(hexs) < 8:  # hex data lines carry >= 8 bytes
            continue
        blob += bytes.fromhex("".join(hexs))
        if len(blob) >= size:
            break
    if len(blob) < size:
        print(f"warning: captured {len(blob)} bytes, expected {size}",
              file=__import__("sys").stderr)
    return blob[:size]


def load_gpucore_segments(path):
    if not path:
        return None
    from decode_gpucore import load_segments
    import mmap
    f = open(path, "rb")
    data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    segs = load_segments(data)
    return data, segs


def locate(segs, val):
    """Return a short segment annotation for a pointer value, or None."""
    if segs is None:
        return None
    for (sva, smsz, soff, sfsz, fl) in segs:
        if sva <= val < sva + smsz:
            readable = "yes" if (val - sva) < sfsz else "NO (unwritten/bss)"
            return (f"in seg {sva:#x} memsz={smsz:#x} "
                    f"rel={(val - sva):#x} readable={readable}")
    return "not in any PT_LOAD"


def decode_tensorinfo(blob, base, cap, index_fmt):
    data = struct.unpack_from("<Q", blob, base)[0]
    sizes = struct.unpack_from("<" + index_fmt * cap, blob, base + 8)[:cap]
    strides = struct.unpack_from(
        "<" + index_fmt * cap, blob, base + 8 + 4 * cap)[:cap]
    dims = struct.unpack_from("<I", blob, base + 8 + 4 * cap + 4 * cap)[0]
    return data, sizes[:dims], strides[:dims], dims


def abi_mode(blob, abi, segs):
    print(f"=== ABI decode: {abi['name']} ({len(blob)} bytes) ===")
    cap = abi.get("tensorinfo_dim_cap", 25)
    ifmt = abi.get("tensorinfo_dtype", "uint")
    index_fmt = "I" if ifmt == "uint" else "i"
    for f in abi["fields"]:
        off = f["offset"]
        if f["type"] == "tensorinfo":
            data, sizes, strides, dims = decode_tensorinfo(blob, off, cap, index_fmt)
            ann = f"  -> {locate(segs, data)}" if segs is not None else ""
            print(f"{f['name']}: data={data:#x} dims={dims} "
                  f"sizes={list(sizes)} strides={list(strides)}{ann}")
        else:
            fmt = SCALAR_FORMATS[f["type"]]
            val = struct.unpack_from(fmt, blob, off)[0]
            ann = f"  -> {locate(segs, val)}" if segs is not None and f["type"] == "u64" else ""
            print(f"{f['name']} ({f['type']}): {val}{ann}")


def raw_mode(blob, segs):
    print(f"=== RAW blind scan ({len(blob)} bytes) ===")
    nw = len(blob) // 8
    print("u64 words (potential pointers / addresses):")
    for i in range(nw):
        off = i * 8
        val = struct.unpack_from("<Q", blob, off)[0]
        ann = ""
        if segs is not None:
            loc = locate(segs, val)
            ann = f"  -> {loc}" if loc else "  (not a gpucore segment base)"
        elif val >= 0x100000000 and val % 8 == 0:
            ann = "  (plausible pointer — pass --gpucore to identify)"
        print(f"  @{off:04d}: {val:#018x}{ann}")
    print("\ni32 / i64 scalar views (offsets 0..min(size,64)):")
    for off in range(0, min(len(blob), 64), 4):
        if off + 4 <= len(blob):
            i = struct.unpack_from("<i", blob, off)[0]
            print(f"  @{off:04d}: i32={i}")


def main():
    ap = argparse.ArgumentParser(
        description="Decode a kernel args hex dump from an HSA crash log "
                    "(generic; --raw for kernels with unknown ABI).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("crash_log")
    ap.add_argument("abi_json", nargs="?", help="custom ABI JSON (omit for default)")
    ap.add_argument("--raw", action="store_true",
                    help="blind scan, no ABI required")
    ap.add_argument("--gpucore", help="gpucore ELF path to cross-ref pointers")
    ap.add_argument("--size", type=int, help="args buffer size (default: from log)")
    args = ap.parse_args()

    lines = read_lines(args.crash_log)
    size = args.size or find_size(lines)
    blob = extract_blob(lines, find_arg_dump_start(lines, size), size)

    data, segs = None, None
    if args.gpucore:
        data, segs = load_gpucore_segments(args.gpucore)

    if args.raw:
        raw_mode(blob, segs)
    else:
        abi = json.load(open(args.abi_json)) if args.abi_json else DEFAULT_ABI
        abi_mode(blob, abi, segs)

    if data is not None:
        data.close()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
