#!/usr/bin/env python3
"""从对标实现（tilelang JIT）推导手写 HIP kernel 的 aicc 编译命令行。

为什么需要它：对标实现的 `-mllvm` 集合是**运行时**按 arch + pass_configs 拼出来的
（`tilelang/contrib/hcu.py: get_hcu_compile_flags`），不是静态列表；其中还有条件项
（`TL_ENABLE_FAST_MATH` → 追加 `-enable-hcu-approx-func-fp-math`）。抄一份固定列表必然漂，
漏一条可能慢 10%+ 而数值完全不变。本工具直接调用同一个函数，保证「镜像」是逐字的。

用法：
  python3 derive_flags.py --arch gfx938 --fast-math
  python3 derive_flags.py --arch gfx938 --fast-math --wdra
  python3 derive_flags.py --arch gfx938 --fast-math --hip dev/x.hip --out dev/libx.so
  python3 derive_flags.py --arch gfx938 --sniff /path/to/ref_kernel.py
  python3 derive_flags.py --arch gfx938 --print-static     # 无 tilelang 时的参考（带警告）

前置：tilelang 源码可导入（设 TILELANG_HYGON_ROOT，或已 pip 安装）；aicc 在 PATH
（`/opt/dtk/bin`），否则 `get_hcu_compile_flags` 会按 DTK 判定直接返回 []。

退出码：0 成功；2 无法导入 tilelang；3 推导出的 flag 为空（多半是 aicc 不在 PATH）。
"""
import argparse
import os
import re
import subprocess
import sys

FAST_MATH_KEY = "tl.enable_fast_math"
WDRA_KEY = "tl.enable_hcu_wdra"

# 无 tilelang 时的参考副本**仅用于对照**，判据永远是上面那个函数。
STATIC_GFX938 = [
    "-mllvm=-enable-latency-hack=true",
    "-mllvm=-mmac-latency=5",
    "-mllvm=-ds-load-store-latency=6",
    "-mllvm=-disable-machine-sink=True",
    "-mllvm=-check-valu-data-forward-hazards=0",
    "-mllvm=-disable-cluster-lds-memops=true",
    "-mllvm=-amdgpu-disable-backoff-barrier=false",
    "-mllvm=-support-768-vgprs=true",
    "-mllvm=-hcu-update-wait-by-reverse-search=true",
    "-mllvm=-hcu-pre-emit-load-store-opt=false",
]


def _import_tilelang():
    root = os.environ.get("TILELANG_HYGON_ROOT")
    if root and os.path.isdir(os.path.join(root, "src")):
        src = os.path.join(root, "src")
        if src not in sys.path:
            sys.path.insert(0, src)
    try:
        from tilelang.contrib.hcu import get_hcu_compile_flags  # noqa: F401
        return get_hcu_compile_flags
    except Exception:
        return None


def sniff_pass_configs(path):
    """从源码里嗅出开了哪些会改变 flag 的 pass_configs（启发式，仍需 token diff 复核）。"""
    text = open(path, encoding="utf-8", errors="ignore").read()
    found = {
        "fast_math": bool(re.search(r"TL_ENABLE_FAST_MATH|enable_fast_math", text)),
        "wdra": bool(re.search(r"TL_ENABLE_HCU_WDRA|enable_hcu_wdra", text)),
    }
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="gfx938")
    ap.add_argument("--fast-math", action="store_true", help="镜像 TL_ENABLE_FAST_MATH（几乎总是要）")
    ap.add_argument("--wdra", action="store_true", help="镜像 TL_ENABLE_HCU_WDRA（实测 +1.2% 更慢，谨慎）")
    ap.add_argument("--sniff", metavar="KERNEL_PY", help="从对标 kernel 源码嗅 pass_configs")
    ap.add_argument("--hip", metavar="SRC.hip", help="给了就顺手编译")
    ap.add_argument("--out", metavar="OUT.so")
    ap.add_argument("--threads", type=int, default=256, help="编译期无用，仅为打印口径")
    ap.add_argument("--print-static", action="store_true", help="只打印静态副本（无 tilelang 时）")
    a = ap.parse_args()

    if a.sniff:
        sniffed = sniff_pass_configs(a.sniff)
        a.fast_math = a.fast_math or sniffed["fast_math"]
        a.wdra = a.wdra or sniffed["wdra"]
        print(f"[sniff] {a.sniff}: fast_math={sniffed['fast_math']} wdra={sniffed['wdra']}", file=sys.stderr)

    if a.print_static:
        flags = list(STATIC_GFX938)
        if a.fast_math:
            flags.append("-mllvm=-enable-hcu-approx-func-fp-math=true")
        print(" ".join(flags))
        return 0

    fn = _import_tilelang()
    if fn is None:
        print("ERROR: 无法导入 tilelang.contrib.hcu。设 TILELANG_HYGON_ROOT=<tilelang-hygon 根>，"
              "或先 pip install tilelang。\n"
              "       没有对标实现就无法「镜像」——不要退回抄静态列表。", file=sys.stderr)
        return 2

    pass_configs = {}
    if a.fast_math:
        pass_configs[FAST_MATH_KEY] = True
    if a.wdra:
        pass_configs[WDRA_KEY] = True

    flags = fn(a.arch, pass_configs)
    if not flags:
        print("ERROR: 推导结果为空。常见原因：aicc 不在 PATH（hcu.py 在 DTK 环境下会因此直接 "
              "return []）。`export PATH=/opt/dtk/bin:$PATH` 后重试。", file=sys.stderr)
        return 3

    print(" ".join(flags))

    if a.hip:
        tilelang_root = os.environ.get("TILELANG_HYGON_ROOT", "")
        out = a.out or os.path.splitext(a.hip)[0] + ".so"
        cmd = [
            "/opt/dtk/bin/aicc", "-x", "hip", f"--offload-arch={a.arch}:sramecc+",
            "-std=c++20", "-O3",
        ]
        if tilelang_root:
            cmd += ["-I", os.path.join(tilelang_root, "src")]
        cmd += flags + [a.hip, "-shared", "-fPIC", "-o", out]
        print("\n$ " + " ".join(cmd), file=sys.stderr)
        subprocess.run(cmd, check=True)
        print(f"[built] {out}  (threads={a.threads})", file=sys.stderr)
        print("下一步：python3 inspect_so.py " + out + f" --threads {a.threads}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
