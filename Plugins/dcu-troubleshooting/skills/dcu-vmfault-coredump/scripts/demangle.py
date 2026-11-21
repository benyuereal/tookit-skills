#!/usr/bin/env python3
"""Demangle an HSA kernel symbol name (it is a C++ mangled name).

Just a wrapper around `c++filt` so the skill workflow has one entry point.
On systems without c++filt, falls back to a regex that strips the common
`_ZN...` prefix and splits template args.

Usage:
    python demangle.py '_ZN2at6native12_GLOBAL__N_121indexSelectSmallIndex...'
"""
import re
import subprocess
import sys


def cppfilt(name):
    try:
        out = subprocess.run(["c++filt", name], capture_output=True, text=True)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except FileNotFoundError:
        pass
    return None


def fallback(name):
    # _ZN...NNNfooIN...EEE  -> strip _Z, unpack nested names
    if name.startswith("_Z"):
        name = name[2:]
    name = re.sub(r"\d+", "", name)  # drop length prefixes (rough)
    return name


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    name = sys.argv[1]
    print(cppfilt(name) or fallback(name))
