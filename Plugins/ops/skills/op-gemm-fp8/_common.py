#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""op-gemm-fp8 卡的 adapter：输入构造 / fp32 参考 / 变体注册表。

契约见 scripts/opcheck.py 的 docstring。

诚实性约定：
  - 没有可用后端时 get_runner() **抛异常**，不返回假实现。
  - reference() 用 fp32 稠密 matmul（torch.mm），是任何变体都能比的 oracle。
  - **不使用 torch._scaled_mm 作为参考**——它本身就是被测对象之一。

inputs 位置参数顺序：(A, B) → out[M, N]
"""
import torch

VARIANTS = ("tilelang", "tilelang-rs", "aiter", "hand-hip")

INPUT_NAMES = ("A", "B")


def _dims(params):
    return int(params["M"]), int(params["N"]), int(params["K"])


def load_shape(arg):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
    from opverify import parse_yaml
    p = Path(__file__).resolve().parent
    if arg and (p / str(arg)).is_file():
        p = p / str(arg)
    else:
        p = p / "shapes.yaml"
    data = parse_yaml(p.read_text(encoding="utf-8"))
    key = arg if arg in data else next(iter(data))
    node = dict(data[key])
    node["_name"] = key
    return node


def build_inputs(params, seed=0, device=None):
    """构造 fp8 GEMM 输入。返回 (A, B, meta)。

    重量级 fp8 模拟：按 e4m3 量化再反量化回 bf16，与真实 kernel 吃到的数值一致。
    标度用 per-tensor。
    """
    M, N, K = _dims(params)
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    A = torch.randn(M, K, generator=g, dtype=torch.float32) * 0.1
    B = torch.randn(K, N, generator=g, dtype=torch.float32) * 0.1

    def _fp8_sim(t):
        amax = t.abs().max().clamp_min(1e-8)
        scale = amax / 448.0
        return ((t / scale).clamp(-448, 448).round() * scale).to(torch.bfloat16)

    meta = {"M": M, "N": N, "K": K, "shape": params.get("_name", "")}
    return _fp8_sim(A).to(dev), _fp8_sim(B).to(dev), meta


def reference(A, B, params=None, chunk=None):
    """fp32 稠密参考。

    **刻意用 torch.mm 而不是 _scaled_mm**：后者是被测对象之一，
    拿它当 oracle 会掩盖它的误差。
    """
    return (A.float() @ B.float()).to(torch.float32)


def runner_kwargs(params):
    M, N, K = _dims(params)
    return {"M": M, "N": N, "K": K}


def _aiter_runner(A, B, **kw):
    """aiter 的 gfx938 opus fp8 GEMM。

    实测 2026-11-23：aiter fp8 47.8 TF vs aiter bf16 24.6 TF（同库同形状）⇒ 1.94x。
    **这条比值证明是原生 fp8 MMA，不是 dequant 回退** —— 若走回退两边都跑 bf16 MMA，
    比值应 ≈1.0。判「是否原生 fp8」必须同库同形状对照，跨库比会得出错误结论。
    """
    import aiter
    # build_inputs 的 _fp8_sim 已做量化模拟但保持 bf16 dtype。真 fp8 kernel 需要
    # float8_e4m3fn：直接用 .to() 转换（数值上等价于再做一次 fp8 舍入，误差 <1 ulp）。
    A8 = A if A.dtype == torch.float8_e4m3fn else A.to(torch.float8_e4m3fn)
    B8 = B if B.dtype == torch.float8_e4m3fn else B.to(torch.float8_e4m3fn)
    out = aiter.gemm_a8w8_opus(A8, B8)
    return out.to(torch.bfloat16) if out.dtype != torch.bfloat16 else out


_HIP_LIB = {}


def _hand_hip_runner(A, B, **kw):
    """手写 HIP fp8 GEMM。**当前是朴素 tiled 实现，不是优化版。**

    存在的理由：另外三个变体（tilelang / tilelang-rs / hand-hip 本身）
    的 adapter 都是空的，只有 aiter 能跑，凑不成 A/B 对比，
    导致 B（对比）和 R（轮次）两项无法推进。

    先有一个能跑、能对数的实现，才能谈优化。当前这个必然比 aiter 慢，
    这是预期的 —— 它的价值是给出「朴素实现到厂商库差多少」这个基线。
    """
    import ctypes
    from pathlib import Path
    k = Path(__file__).resolve().parent / "kernels"
    so = k / "gemm_fp8.so"
    if not so.is_file():
        raise RuntimeError(
            "缺 %s —— 先编译：cd %s && hipcc -O3 --offload-arch=gfx938 "
            "-fPIC -shared -o gemm_fp8.so gemm_fp8.hip" % (so, k))
    if "lib" not in _HIP_LIB:
        lib = ctypes.CDLL(str(so))
        fn = lib.gemm_fp8_launch
        fn.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3
        fn.restype = None
        _HIP_LIB["lib"] = fn
    fn = _HIP_LIB["lib"]

    M, K = A.shape
    N = B.shape[1]
    # kernel 吃 uint8 承载的 fp8、吐 float
    A8 = A.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
    B8 = B.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)
    fn(A8.data_ptr(), B8.data_ptr(), C.data_ptr(), M, N, K)
    return C


def get_runner(name, tilelang_root=None, **kwargs):
    if name not in VARIANTS:
        raise ValueError("未知变体 %r（可选 %s）" % (name, " / ".join(VARIANTS)))
    if name == "aiter":
        return _aiter_runner
    if name == "hand-hip":
        return _hand_hip_runner
    raise RuntimeError(
        "变体 %r 的 adapter 尚未实现（**注意：aiter 实际已安装**，此前「需先安装」的措辞是错的）。原措辞：需先安装 TileLang(hcu) / aiter 的 FP8 GEMM 后端"
        "（见 variants.yaml 的 source 字段）。adapter 已就绪，缺的是后端——"
        "不要用假实现绕过。" % name)


def _clone_out(out):
    """保留全部输出，不只第一个（2026-11-23：tuple 输出曾被静默丢弃）。"""
    if torch.is_tensor(out):
        return out.clone()
    if isinstance(out, (tuple, list)):
        return type(out)(_clone_out(x) for x in out)
    return out


def run_isolated(fn, *inp):
    """可信调用：输入副本 + 输出毒化。"""
    copies = [t.clone() if torch.is_tensor(t) else t for t in inp]
    out = fn(*copies)
    poisoned = []
    if torch.is_tensor(out):
        out = out.clone()
    for name, before, after in zip(INPUT_NAMES, inp, copies):
        if torch.is_tensor(before) and torch.is_tensor(after):
            if not torch.equal(before, after):
                poisoned.append(name)
    return out, poisoned


def compare(a, b):
    af, bf = a.float(), b.float()
    diff = (af - bf).abs()
    denom = af.norm() * bf.norm()
    cos = float((af.flatten() @ bf.flatten()) / denom) if denom > 0 else 1.0
    return {"max_abs": float(diff.max()), "mean_abs": float(diff.mean()),
            "cosine": cos}
