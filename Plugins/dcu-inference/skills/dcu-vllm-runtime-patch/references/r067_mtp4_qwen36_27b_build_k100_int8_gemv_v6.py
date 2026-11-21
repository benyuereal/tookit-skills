#!/usr/bin/env python3
"""Build the validated K100AI/gfx928 native INT8 GEMV used by R026 no-MTP."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "k100_int8_gemv_v6.hip"
BUILD = ROOT / "build_gemv_v6"
OUTPUT = ROOT / "k100_int8_gemv_v6.so"
BUILD.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx928")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(BUILD))

mod = load(
    name="k100_int8_gemv_v6",
    sources=[str(SRC)],
    extra_cflags=["-O3", "-std=c++20"],
    extra_cuda_cflags=["-O3", "-std=c++20", "-fno-gpu-rdc"],
    with_cuda=True,
    verbose=True,
)
shutil.copy2(mod.__file__, OUTPUT)
print(OUTPUT)
