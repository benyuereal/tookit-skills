#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""op-norm-act-swiglu 的 adapter（bootstrap 生成骨架 + 人工补齐）。

    xf      = x + residual                                  (读 x, r)
    h       = w * xf / sqrt(mean(xf^2, -1) + eps)            (读 w；归约沿 H)
    gate,up = h[:, :I], h[:, I:]                             (I = H/2)
    out     = silu(gate) * up                                (写 out, I 宽)

流量：读 x + 读 r + 读 w + 写 out(I宽) = 2H + 0.5H = 2.5H
     M=8192 H=4096 ⇒ 167.8 MB
基线（两步法）：读x+读r+写x+写r + 读h+写out = 4H + 1.5H = 5.5H ⇒ 369.1 MB
    ⇒ 融合把流量降到 45.5%，这是全部收益来源。
"""
import ctypes
from pathlib import Path

import torch

INPUT_NAMES = ("x", "residual", "weight")
EPS = 1e-6
KERNELS = Path(__file__).resolve().parent / "kernels"

VARIANTS = ("ref-heavy", "two-step", "fused-hip-mr2")


def _dims(params):
    H = int(params["H"])
    return int(params["M"]), H, H // 2


def load_shape(arg):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
    from opverify import parse_yaml
    p = Path(__file__).resolve().parent
    target = (p / str(arg)) if (arg and (p / str(arg)).is_file()) else (p / "shapes.yaml")
    data = parse_yaml(target.read_text(encoding="utf-8"))
    key = arg if arg in data else next(iter(data))
    node = dict(data[key])
    node["_name"] = key
    return node


def build_inputs(params, seed=0, device=None):
    M, H, I = _dims(params)
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(M, H, generator=g, dtype=torch.float32)
    r = torch.randn(M, H, generator=g, dtype=torch.float32) * 0.1
    w = torch.randn(H, generator=g, dtype=torch.float32) * 0.1 + 1.0
    meta = {"M": M, "H": H, "I": I, "shape": params.get("_name", ""),
            "bytes_fused": int((2 * M * H + M * I) * 2),
            "bytes_twostep": int((4 * M * H + M * H + M * I) * 2)}
    return (x.to(torch.bfloat16).to(dev), r.to(torch.bfloat16).to(dev),
            w.to(torch.bfloat16).to(dev), meta)


def reference(x, residual, weight, params=None, chunk=None):
    """fp32 参考。eps 在 sqrt **内部**（与 lightop 的 fused_add_rms_norm 一致）。"""
    I = x.shape[1] // 2
    xf = x.float() + residual.float()
    h = xf / torch.sqrt(xf.pow(2).mean(dim=-1, keepdim=True) + EPS)
    h = h * weight.float()
    return (torch.nn.functional.silu(h[:, :I]) * h[:, I:]).to(x.dtype)


def runner_kwargs(params):
    M, H, I = _dims(params)
    return {"M": M, "H": H, "I": I}


_LIB = {}


def _hip_runner(x, r, w, **kw):
    """手写 HIP 融合 kernel（2 行/CTA）。只在 H==4096 且 M 任意时可用。"""
    H = x.shape[1]
    if H != 4096:
        raise ValueError(
            "fused-hip-mr2 按 H=4096 / THREADS=256 编译（THREADS*8 == I=2048）。"
            "收到 H=%d —— 换形状必须重新生成。" % H)
    if "lib" not in _LIB:
        so = KERNELS / "mr2.so"
        if not so.is_file():
            raise RuntimeError("缺 %s —— 先跑 kernels/build.sh" % so)
        lib = ctypes.CDLL(str(so))
        fn = lib.norm_act_swiglu_mr_launch
        fn.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int, ctypes.c_int,
                                               ctypes.c_float]
        fn.restype = None
        _LIB["lib"] = fn
    M = x.shape[0]
    out = torch.empty(M, H // 2, dtype=x.dtype, device=x.device)
    _LIB["lib"](x.data_ptr(), r.data_ptr(), w.data_ptr(), out.data_ptr(),
                M, H, float(kw.get("eps", EPS)))
    return out


def _twostep_runner(x, r, w, **kw):
    """两步法基线（lightop 现成路径）。**原地改写 x 与 residual**。"""
    import lightop
    o1, o2 = lightop.fused_add_rms_norm(x, r, w, EPS)
    out = torch.empty(x.shape[0], x.shape[1] // 2, dtype=x.dtype, device=x.device)
    lightop.fuse_silu_and_mul(o1, out)
    return out


def get_runner(name, tilelang_root=None, **kwargs):
    if name not in VARIANTS:
        raise ValueError("未知变体 %r（可选 %s）" % (name, " / ".join(VARIANTS)))
    if name == "ref-heavy":
        return lambda x, r, w, **kw: reference(x, r, w)
    if name == "two-step":
        return _twostep_runner
    if name == "fused-hip-mr2":
        return _hip_runner
    raise RuntimeError("变体 %r 未实现" % name)


def run_isolated(fn, *inp):
    copies = [t.clone() if torch.is_tensor(t) else t for t in inp]
    out = fn(*copies)
    out = out.clone() if torch.is_tensor(out) else out
    poisoned = [n for n, a, b in zip(INPUT_NAMES, inp, copies)
                if torch.is_tensor(a) and torch.is_tensor(b) and not torch.equal(a, b)]
    return out, poisoned


def compare(a, b):
    af, bf = a.float(), b.float()
    d = (af - bf).abs()
    denom = af.norm() * bf.norm()
    cos = float((af.flatten() @ bf.flatten()) / denom) if denom > 0 else 1.0
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()), "cosine": cos}
