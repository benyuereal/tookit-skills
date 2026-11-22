#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""op-fused-add-rmsnorm 卡的 adapter。

契约见 scripts/opcheck.py。inputs 顺序：(x, residual, weight) → y

**本卡的特殊性**：它是校准件。字节数完全确定（3*M*H*2B），
所以带宽达成率的分母没有争议——这使它成为验证判据本身的最好载体。

现成实现已核实可 import：lightop.RMSNormAdd（同族 RMSNormAddDropout /
RMSNormAddDropoutBias）。库名用 libraries-registry.yaml 的 canonical id。
"""
import torch

VARIANTS = ("ref-heavy", "lightop-rmsnormadd", "aiter-norm")

INPUT_NAMES = ("x", "residual", "weight")

EPS = 1e-6


def _dims(params):
    return int(params["M"]), int(params["H"])


def load_shape(arg):
    import sys
    from pathlib import Path
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
    """构造 (x, residual, weight)。返回 (x, r, w, meta)。"""
    M, H = _dims(params)
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.randn(M, H, generator=g, dtype=torch.float32)
    r = torch.randn(M, H, generator=g, dtype=torch.float32) * 0.1
    w = torch.randn(H, generator=g, dtype=torch.float32) * 0.1 + 1.0

    meta = {"M": M, "H": H, "shape": params.get("_name", ""),
            # 融合后：读 x + 读 r + 读 w + 写 y
            "bytes_fused": 3 * M * H * 2,
            "bytes_unfused": 4 * M * H * 2}
    return (x.to(torch.bfloat16).to(dev), r.to(torch.bfloat16).to(dev),
            w.to(torch.bfloat16).to(dev), meta)


def reference(x, residual, weight, params=None, chunk=None):
    """fp32 参考：残差先加，再在 sqrt 内加 eps。

    与 lightop.RMSNormAdd 的口径一致——这是 contract.yaml 的
    eps_position / residual_semantics 两条不变量要求的。
    """
    xf = x.float() + residual.float()
    rms = torch.sqrt(xf.pow(2).mean(dim=-1, keepdim=True) + EPS)
    return (xf / rms) * weight.float()


def runner_kwargs(params):
    M, H = _dims(params)
    return {"M": M, "H": H}


def get_runner(name, tilelang_root=None, **kwargs):
    if name not in VARIANTS:
        raise ValueError("未知变体 %r（可选 %s）" % (name, " / ".join(VARIANTS)))
    if name == "ref-heavy":
        def _ref(x, r, w, **kw):
            return reference(x, r, w, params=kw)
        return _ref
    if name == "lightop-rmsnormadd":
        # 签名已核对（实测 2026-11-22，lightop 0.6.0+dtk2604）：
        #   lightop.RMSNormAdd(normalized_shape, eps=1e-5, inplace=False) 是 nn.Module
        #   forward(X, residual) -> y；weight 是模块属性
        # 核对方式不是猜：import 后看 inspect.signature 与 forward 源码。
        try:
            import lightop
        except ImportError as e:
            raise RuntimeError("lightop 不可 import: %s" % e)
        if not hasattr(lightop, "RMSNormAdd"):
            raise RuntimeError("lightop.RMSNormAdd 不存在（期望 0.6.0+dtk2604）")

        H = int(kwargs.get("H", 0))
        if not H:
            raise RuntimeError("get_runner 需要 H（归一化维）才能构造 RMSNormAdd")
        mod = lightop.RMSNormAdd(H, eps=EPS).cuda().to(torch.bfloat16)

        def _run(x, r, w, **_kw):
            with torch.no_grad():
                mod.weight.copy_(w.to(mod.weight.dtype))
                return mod(x.to(torch.bfloat16), r.to(torch.bfloat16))
        return _run
    if name == "aiter-norm":
        raise RuntimeError(
            "变体 aiter-norm 需核对 aiter 的 norm 族调用签名后接入；"
            "本卡当前以 lightop-rmsnormadd 为首选，aiter 仅作非融合对照。")
def run_isolated(fn, *inp):
    copies = [t.clone() if torch.is_tensor(t) else t for t in inp]
    out = fn(*copies)
    out = out.clone() if torch.is_tensor(out) else out
    poisoned = []
    for name, before, after in zip(INPUT_NAMES, inp, copies):
        if torch.is_tensor(before) and torch.is_tensor(after) and not torch.equal(before, after):
            poisoned.append(name)
    return out, poisoned


def compare(a, b):
    af, bf = a.float(), b.float()
    diff = (af - bf).abs()
    denom = af.norm() * bf.norm()
    cos = float((af.flatten() @ bf.flatten()) / denom) if denom > 0 else 1.0
    return {"max_abs": float(diff.max()), "mean_abs": float(diff.mean()), "cosine": cos}


def achieved_bandwidth(median_ms, meta):
    """带宽达成率。分母固定用实测峰值 1724 GB/s，不用标称 1100。"""
    peak_gbs = 1724.0
    gbs = meta["bytes_fused"] / (median_ms * 1e-3) / 1e9
    return {"achieved_gbs": gbs, "peak_gbs": peak_gbs, "ratio": gbs / peak_gbs}
