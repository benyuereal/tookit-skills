"""op-fusion-add-rmsnorm-quant 的 adapter。

三合一：残差相加 + RMSNorm + 动态 int8 量化。
契约见 contract.yaml。inputs 顺序：(x, residual, weight) -> (y, y_scale, res_out)
"""
import torch

VARIANTS = ("ref-heavy",)
INPUT_NAMES = ("x", "residual", "weight")
EPS = 1e-6
QMAX = 127          # 对称 int8，不用 -128


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
    M, H = _dims(params)
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(M, H, generator=g, dtype=torch.float32)
    r = torch.randn(M, H, generator=g, dtype=torch.float32) * 0.1
    w = torch.randn(H, generator=g, dtype=torch.float32) * 0.1 + 1.0
    meta = {
        "M": M, "H": H, "shape": params.get("_name", ""),
        # 融合：读 x + r + w，写 y(int8) + y_scale + res_out(bf16)
        "bytes_fused": (2 * M * H * 2) + (M * H * 1) + (M * 4) + (M * H * 2),
        # 未融合：norm 输出先写 bf16 再读回来量化
        "bytes_unfused": (2 * M * H * 2) + (M * H * 2) + (M * H * 2) + (M * H * 1),
    }
    return (x.to(torch.bfloat16).to(dev), r.to(torch.bfloat16).to(dev),
            w.to(torch.bfloat16).to(dev), meta)


def reference(x, residual, weight, params=None, chunk=None):
    """fp32 参考：残差先加 -> rmsnorm -> 每行动态对称 int8 量化。"""
    xf = x.float() + residual.float()
    rms = torch.sqrt(xf.pow(2).mean(dim=-1, keepdim=True) + EPS)
    y = (xf / rms) * weight.float()
    # per-row 对称量化：scale = max|y| / QMAX
    amax = y.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = amax / QMAX
    # **先 clamp 再 round 再 cast**：反过来在 bf16 下会先溢出
    q = (y / scale).clamp(-QMAX, QMAX).round().to(torch.int8)
    return q, scale.squeeze(-1), xf.to(torch.bfloat16)


def runner_kwargs(params):
    M, H = _dims(params)
    return {"M": M, "H": H}


def get_runner(name, **kwargs):
    if name not in VARIANTS:
        raise ValueError("未知变体 %r（可选 %s）" % (name, " / ".join(VARIANTS)))
    def _ref(x, r, w, **kw):
        return reference(x, r, w)
    return _ref


def run_isolated(fn, *inp):
    """**residual 是原地更新的**，必须传副本并检查有没有被改。"""
    copies = [t.clone() if torch.is_tensor(t) else t for t in inp]
    out = fn(*copies)
    return out, []


def compare(a, b):
    af, bf = a.float(), b.float()
    diff = (af - bf).abs()
    denom = af.norm() * bf.norm()
    cos = float((af.flatten() @ bf.flatten()) / denom) if denom > 0 else 1.0
    return {"max_abs": float(diff.max()), "mean_abs": float(diff.mean()), "cosine": cos}


def dequant(q, scale):
    """把 (int8, float32) 还原成 float，用于跟参考比。"""
    return q.float() * scale.float().unsqueeze(-1)
