#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""手写 HIP kernel 的最小加载链路（编译 + ctypes 调用）。

## 为什么有这个文件

`op-hip-compile` 讲的是「编译调优」，但没有「从零把第一个 HIP kernel 跑起来」的模板。
结果是 agent 第一次写 HIP 时要自己解决编译 + 加载 + stream 匹配三件事，
而这三件事各自都能卡住人。本文件把它压成 20 行。

## 用法

    from load import load_kernel
    fn = load_kernel("fused_add_rmsnorm_reg.hip", entry="fused_rms_launch")
    fn(x.data_ptr(), r.data_ptr(), w.data_ptr(), y.data_ptr(), ro.data_ptr(),
       M, H, 1e-6)

## 三条必须知道的事（都是本仓踩出来的）

1. **编译期常量必须走 -D，不能运行期传**。`VEC` 若作为函数参数，
   `xv[]/rv[]` 会计出到私有内存（实测 private 0→528B，慢 3.3 倍，且不报错）。
2. **`private` 必须为 0**。溢出到私有内存的 kernel 跑得动、算得对、就是慢 ——
   这种退化在跑分里看起来只是「这个方向不行」，会让人把对的改法判成错的。
3. **stream 用默认流**。kernel 里的 `<<<grid, block>>>` 走默认流，与 torch 的
   默认流一致；要换流必须显式传，不要假设。
"""
import ctypes
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARCH = "gfx938"


def compile_kernel(src, so=None, defines=None, force=False):
    """.hip → .so。defines 走 -D（编译期常量，见上文第 1 条）。"""
    src, so = Path(src), Path(so or Path(src).with_suffix(".so"))
    if so.is_file() and not force and so.stat().st_mtime > src.stat().st_mtime:
        return so
    cmd = ["hipcc", "-O3", "--offload-arch=%s" % ARCH, "-fPIC", "-shared",
           "-o", str(so), str(src)]
    for k, v in (defines or {}).items():
        cmd += ["-D%s=%s" % (k, v)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError("hipcc 失败：\n%s\n%s" % (p.stdout, p.stderr))
    return so


def inspect_so(so):
    """读编译产物的资源画像。**private 必须为 0**。"""
    spec = HERE.parents[1] / "op-hip-compile/inspect_so.py"
    if not spec.is_file():
        return None
    p = subprocess.run([sys.executable, "-I", str(spec), str(so), "--json"],
                       capture_output=True, text=True)
    if p.returncode != 0:
        return None
    import json
    ks = json.loads(p.stdout).get("kernels") or []
    return ks[0] if ks else None


def read_defines(src):
    """从源码读 `#define NAME val` —— 用来校验 Python 侧的假设与 kernel 里的一致。

    这个函数存在的理由：kernel 里的 THREADS/VEC 是**生成时固化**的，
    Python 侧按 H == THREADS*8*VEC 做前置检查。两者若不一致，
    检查放过了、kernel 却会越界 —— 而且不报错，只算错。
    """
    import re
    out = {}
    for m in re.finditer(r"(?m)^#define\s+(THREADS|VEC|WARPS)\s+(\d+)",
                         Path(src).read_text(encoding="utf-8")):
        out[m.group(1)] = int(m.group(2))
    return out


def load_kernel(src, entry, defines=None, force=False, check_health=True):
    """编译并返回可调用的 ctypes 函数。

    check_health=True 时若 private>0 直接抛 —— 见上文第 2 条。
    """
    so = compile_kernel(src, defines=defines, force=force)
    if check_health:
        prof = inspect_so(so)
        if prof and int(prof.get("private") or 0) > 0:
            raise RuntimeError(
                "%s 的 private=%dB > 0：寄存器数组溢出到私有内存，慢 3.3 倍且不报错。"
                "常见原因：数组索引用了运行期变量（如 `for j < vec` 而非 `j < VEC`）。"
                % (Path(src).name, prof["private"]))
    lib = ctypes.CDLL(str(so))
    fn = getattr(lib, entry)
    fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int, ctypes.c_int,
                                           ctypes.c_float]
    fn.restype = None
    return fn


if __name__ == "__main__":
    # 自检：编译本目录的 kernel 并打印资源画像
    for hip in sorted(HERE.glob("*.hip")):
        so = compile_kernel(hip)
        prof = inspect_so(so)
        print("%-32s %s" % (hip.name, prof))
