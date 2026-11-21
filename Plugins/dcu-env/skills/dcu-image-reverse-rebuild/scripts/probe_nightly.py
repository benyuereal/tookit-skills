#!/usr/bin/env python3
"""Probe whether a DCU wheel version is REALLY available on sourcefind/nightly/dtk.

Usage: python3 probe_nightly.py <package> <version> [index-url]
Prints the matching version line. A version is "real" only if the line does NOT
contain the 'INSTALLED' prefix (that prefix means it's the locally-installed
version of the probed container, not downloadable from the index).

Example:
  python3 probe_nightly.py flash_mla 1.2.0+das.opt1.dtk2604.torch290.2605281449.ge3aed6
"""
import sys, subprocess

PKG = sys.argv[1]
VER = sys.argv[2]
IDX = sys.argv[3] if len(sys.argv) > 3 else "https://pypi.sourcefind.cn/nightly/dtk/"

out = subprocess.run(
    ["pip", "index", "versions", PKG, "-i", IDX],
    capture_output=True, text=True,
).stdout

for line in out.splitlines():
    if VER in line:
        if "INSTALLED" in line:
            print(f"[NOT-AVAILABLE-AS-DOWNLOAD] {line.strip()}  (INSTALLED only)")
        else:
            print(f"[AVAILABLE] {line.strip()}")
        break
else:
    print(f"[MISSING] {VER} not found in index output:\n{out[:500]}")
