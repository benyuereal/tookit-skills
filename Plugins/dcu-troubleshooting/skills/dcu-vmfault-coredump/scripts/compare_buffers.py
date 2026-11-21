#!/usr/bin/env python3
"""Byte-level verification that a faulting kernel's destination rows were
copied from a source buffer at the given indices (fault-time data proof).

For an `index_select`-style OOB crash (e.g. `embed_tokens` embedding a bad
token id), read from a gpucore ELF:
  - index tensor (int64[N]): the token/row ids the kernel was processing
  - src (nrow x ncol row-major): the lookup table (e.g. embed weight)
  - dst (N x ncol): the output
and check every dst row i (with a non-negative index) byte-matches
src row index[i]. A dst row whose index was the OOB culprit is typically
unwritten (NaN / stale), which is itself the confirmation that the kernel
faulted exactly there.

Usage:
    python compare_buffers.py gpucore.21292 \
        --index 0x7ef3cd1e3400:4 \
        --src 0x7ef1cb000000:248320:2048 \
        --dst 0x7ef3247aaa00:4:2048 \
        [--dtype float16|float32|int32|int64|int8|uint8] \
        [--tolerance N]        # compare only the first N bytes per row
"""
import argparse
import struct

from decode_gpucore import load_segments, read_va

DTYPE_SIZE = {"float16": 2, "float32": 4, "int32": 4, "int64": 8,
              "int8": 1, "uint8": 1}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gpucore")
    ap.add_argument("--index", required=True, help="index tensor VA[:count] (int64)")
    ap.add_argument("--src", required=True, help="src buffer VA:nrow:ncol")
    ap.add_argument("--dst", required=True, help="dst buffer VA:nrow:ncol")
    ap.add_argument("--dtype", default="float16",
                    choices=sorted(DTYPE_SIZE), help="element dtype (default float16)")
    ap.add_argument("--tolerance", type=int, default=0,
                    help="compare only first N bytes of each row (default all)")
    args = ap.parse_args()

    import mmap
    with open(args.gpucore, "rb") as f:
        data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    segs = load_segments(data)

    def va3(spec, what):
        parts = spec.split(":")
        va = int(parts[0], 0)
        n = [int(x) for x in parts[1:]]
        return va, n

    idx_va, idx_n = va3(args.index, "index")
    idx_n = idx_n[0]
    src_va, (src_nrow, src_ncol) = va3(args.src, "src")
    dst_va, (dst_nrow, dst_ncol) = va3(args.dst, "dst")
    esz = DTYPE_SIZE[args.dtype]

    idx_chunk, idx_info = read_va(data, segs, idx_va, idx_n * 8)
    if idx_chunk is None or len(idx_chunk) < idx_n * 8:
        print(f"index read failed: {idx_info} (got {0 if idx_chunk is None else len(idx_chunk)}B)")
        return 1
    indices = list(struct.unpack_from("<" + "q" * idx_n, idx_chunk))
    print(f"index[{idx_n}] = {indices}   ({idx_info})")

    rowsz = dst_ncol * esz
    cmp_n = rowsz if args.tolerance <= 0 else min(rowsz, args.tolerance)

    ok = 0
    for i in range(dst_nrow):
        idx = indices[i] if i < len(indices) else None
        dst_row, d_info = read_va(data, segs, dst_va + i * rowsz, cmp_n)
        if idx is not None and 0 <= idx < src_nrow:
            src_row, s_info = read_va(data, segs, src_va + idx * rowsz, cmp_n)
            if dst_row is None or src_row is None or len(dst_row) < cmp_n or len(src_row) < cmp_n:
                print(f"dst row {i}: unreadable (idx={idx}) [{d_info} / {s_info}]")
                continue
            match = dst_row[:cmp_n] == src_row[:cmp_n]
            ok += bool(match)
            print(f"dst row {i} vs src row {idx}: {'MATCH' if match else 'MISMATCH'} "
                  f"({cmp_n}/{cmp_n} bytes)")
        else:
            print(f"dst row {i}: index={idx} -> OOB/invalid (expected unwritten/stale dst)")
    print(f"\n{ok}/{dst_nrow} dst rows byte-match src at their index")
    return 0 if ok == dst_nrow else 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
