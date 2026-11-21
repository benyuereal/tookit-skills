#!/usr/bin/env python3
"""Read memory from an AMD Hydra gpucore ELF (HSA_ENABLE_COREDUMP=1 snapshot)
at specific GPU virtual addresses.

The gpucore is a full-ELF memory dump. GPU device pointers from a decoded
kernel-args buffer (see parse_kernel_args.py) are virtual addresses into its
PT_LOAD segments, so you can read the fault-time values directly.

Usage:
    python decode_gpucore.py gpucore.21292 VA[:nbytes] [VA[:nbytes] ...]
    python decode_gpucore.py gpucore.21292 --va-list file.txt
      (one 'VA[:nbytes]' per line; comments with # allowed)

Examples:
    python decode_gpucore.py gpucore.21292 0x7ef3cd1e3400:64  0x7ef1cb000000:96
    python decode_gpucore.py gpucore.21292 0x7ef3cd1e3400       # default 64 bytes
    python decode_gpucore.py gpucore.21292 --ptrs 0x7ef3cd1e3400 0x7ef1cb000000
      # attribution: which segment each pointer belongs to + readable bytes
"""
import mmap
import struct
import sys


def load_segments(data):
    assert data[:4] == b"\x7fELF", "not an ELF file"
    e_type, e_machine = struct.unpack_from("<HH", data, 16)
    e_phoff = struct.unpack_from("<Q", data, 32)[0]
    e_phentsize, e_phnum = struct.unpack_from("<HH", data, 54)
    segs = []
    for i in range(e_phnum):
        off = e_phoff + i * e_phentsize
        p_type, p_flags = struct.unpack_from("<II", data, off)
        p_offset, p_vaddr = struct.unpack_from("<QQ", data, off + 8)
        p_filesz, p_memsz = struct.unpack_from("<QQ", data, off + 32)
        if p_type == 1:  # PT_LOAD
            segs.append((p_vaddr, p_memsz, p_offset, p_filesz, p_flags))
    print(f"ELF type={e_type} machine={e_machine:#x} PT_LOAD segments: {len(segs)}")
    for va, msz, foff, fsz, fl in sorted(segs):
        print(f"  seg va={va:#x} memsz={msz:#x} foff={foff:#x} filesz={fsz:#x} flags={fl}")
    return segs


def read_va(data, segs, va, nbytes):
    """Read from a memory-mapped gpucore. `data` is an mmap; returns bytes.

    memory may extend past file size (bss) — clamp to the written extent.
    """
    for (sva, smsz, soff, sfsz, _fl) in segs:
        if sva <= va < sva + smsz:
            rel = va - sva
            if rel >= sfsz:
                return None, f"VA {va:#x} within seg but beyond filesz (unwritten/bss)"
            fstart = soff + rel
            return bytes(data[fstart:fstart + min(nbytes, sfsz - rel)]), \
                f"seg va={sva:#x} rel={rel:#x}"
    return None, f"VA {va:#x} not in any PT_LOAD"


def parse_targets(argv):
    targets = []
    for a in argv:
        if ":" in a:
            va_s, n_s = a.split(":", 1)
            nbytes = int(n_s, 0)
        else:
            va_s, nbytes = a, 64
        targets.append((int(va_s, 0), nbytes))
    return targets


def ptrs_mode(path, ptrs):
    """Report which PT_LOAD segment each pointer belongs to (buffer attribution)."""
    with open(path, "rb") as f:
        data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    segs = load_segments(data)
    print("\n=== pointer attribution ===")
    for p in ptrs:
        hit = None
        for (sva, smsz, soff, sfsz, fl) in segs:
            if sva <= p < sva + smsz:
                rel = p - sva
                readable = (rel < sfsz)
                hit = (sva, smsz, rel, sfsz, readable)
                break
        if hit:
            sva, smsz, rel, sfsz, readable = hit
            print(f"{p:#x}: seg {sva:#x} memsz={smsz:#x} rel={rel:#x} "
                  f"readable={readable} (bytes-to-end={sfsz - rel if readable else 0})")
        else:
            print(f"{p:#x}: NOT in any PT_LOAD (host/GTT/other address space)")
    return 0


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    path = sys.argv[1]
    rest = sys.argv[2:]
    if rest and rest[0] == "--va-list":
        targets = []
        for line in open(rest[1]):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            va_s, _, n_s = line.partition(":")
            targets.append((int(va_s, 0), int(n_s, 0) if n_s else 64))
    elif rest and rest[0] == "--ptrs":
        ptrs = [int(p, 0) for p in rest[1:]]
        return ptrs_mode(path, ptrs)
    else:
        targets = parse_targets(rest)

    with open(path, "rb") as f:
        data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    segs = load_segments(data)

    for va, nbytes in targets:
        chunk, info = read_va(data, segs, va, nbytes)
        print(f"\n=== VA {va:#x} ({nbytes} bytes) -> {info} ===")
        if chunk is None:
            print("  (not available)")
            continue
        n = len(chunk)
        if n >= 8:
            vals = struct.unpack_from("<" + "q" * (n // 8), chunk)
            print(f"  int64[{n // 8}]: {list(vals)}")
        if n >= 4:
            vals = struct.unpack_from("<" + "f" * (n // 4), chunk)
            print(f"  float32[{n // 4}]: {list(vals)}")
        print("  hex:", chunk.hex())
    return 0


if __name__ == "__main__":
    sys.exit(main())
