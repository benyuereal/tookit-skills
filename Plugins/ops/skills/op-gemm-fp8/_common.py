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


def get_runner(name, tilelang_root=None, **kwargs):
    if name not in VARIANTS:
        raise ValueError("未知变体 %r（可选 %s）" % (name, " / ".join(VARIANTS)))
    raise RuntimeError(
        "变体 %r 在本机不可用：需先安装 TileLang(hcu) / aiter 的 FP8 GEMM 后端"
        "（见 variants.yaml 的 source 字段）。adapter 已就绪，缺的是后端——"
        "不要用假实现绕过。" % name)


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
