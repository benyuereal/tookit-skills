#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""op-rmsnorm-fused 卡的 adapter。

契约见 scripts/opcheck.py。inputs 顺序：(x, weight, residual) → y

本卡是最典型的 bandwidth-bound 算子，字节数完全确定：
  不融合残差：读 x + 读 w + 写 y = 2*M*H*2B
  融合残差  ：读 x + 读 r + 读 w + 写 y = 3*M*H*2B
所以带宽达成率的分母没有争议——这使它成为校准判据的最佳载体。

诚实性约定：没有后端就抛，不返回假实现。
"""
import torch

VARIANTS = ("ref-heavy", "lightop-fused", "torch-native")

INPUT_NAMES = ("x", "weight", "residual")


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
    """构造 (x, weight, residual)。返回 (x, w, r, meta)。"""
    M, H = _dims(params)
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.randn(M, H, generator=g, dtype=torch.float32)
    w = torch.randn(H, generator=g, dtype=torch.float32) * 0.1 + 1.0
    r = torch.randn(M, H, generator=g, dtype=torch.float32) * 0.1

    meta = {"M": M, "H": H, "shape": params.get("_name", ""),
            "bytes_unfused": 2 * M * H * 2, "bytes_fused": 3 * M * H * 2}
    return (x.to(torch.bfloat16).to(dev), w.to(torch.bfloat16).to(dev),
            r.to(torch.bfloat16).to(dev), meta)


def reference(x, weight, residual, params=None, chunk=None):
    """fp32 参考：融合残差的 RMSNorm。

    eps 加在 sqrt **内部**（sqrt(mean+eps)）——这与主流实现一致，
    且是 contract.yaml 的 eps_position 不变量要求的。
    """
    eps = 1e-6
    xf = x.float() + residual.float()          # 先融合残差
    rms = torch.sqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf / rms) * weight.float()


def runner_kwargs(params):
    M, H = _dims(params)
    return {"M": M, "H": H}


def get_runner(name, tilelang_root=None, **kwargs):
    if name not in VARIANTS:
        raise ValueError("未知变体 %r（可选 %s）" % (name, " / ".join(VARIANTS)))
    if name == "ref-heavy":
        def _ref(x, w, r, **kw):
            return reference(x, w, r, params=kw)
        return _ref
    raise RuntimeError(
        "变体 %r 在本机不可用：需先安装 lightop 的融合 norm 实现。"
        "adapter 已就绪，缺的是后端——不要用假实现绕过。" % name)


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


def achieved_bandwidth(median_ms, meta, fused=True):
    """带宽达成率。分母固定用实测峰值，不用标称。"""
    peak_gbs = 1724.0
    nbytes = meta["bytes_fused"] if fused else meta["bytes_unfused"]
    gbs = nbytes / (median_ms * 1e-3) / 1e9
    return {"achieved_gbs": gbs, "peak_gbs": peak_gbs, "ratio": gbs / peak_gbs}
