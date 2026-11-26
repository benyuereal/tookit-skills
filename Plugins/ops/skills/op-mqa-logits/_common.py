"""op-mqa-logits 的 adapter。

DSA indexer 打分：logits[i,j] = sum_h relu(q[i,h,:] . k[j,:]) * w[i,h]
契约见 contract.yaml。inputs 顺序：(q, k, w) -> logits
"""
import torch

VARIANTS = ("ref-heavy",)
INPUT_NAMES = ("q", "k", "w")


def _dims(params):
    return (int(params["B"]), int(params["L"]), int(params["H"]),
            int(params["D"]), int(params["S"]))


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
    """构造 (q, k, w)。返回 (q, k, w, meta)。

    **K 用 randn 生成后会经过 bf16 舍入** —— 这样 reference 和候选
    吃到的数值完全一致（不是 reference 吃 fp32、候选吃 bf16）。
    """
    B, L, H, D, S = _dims(params)
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    q = torch.randn(B, L, H, D, generator=g, dtype=torch.float32) * 0.1
    k = torch.randn(S, H, D, generator=g, dtype=torch.float32) * 0.1
    # W 是 per-(b,l,h) 的标量，正数（打分权重）
    w = torch.rand(B, L, H, generator=g, dtype=torch.float32) + 0.5

    bf = lambda x: x.to(torch.bfloat16).to(dev)          # noqa: E731
    meta = {
        "B": B, "L": L, "H": H, "D": D, "S": S,
        "shape": params.get("_name", ""),
        # 读 q + k + w，写 logits(fp32)
        "bytes_fused": (B * L * H * D + S * H * D) * 2 + B * L * H * 4 + B * L * S * 4,
        "bytes_unfused": (B * L * H * D + S * H * D) * 2 + B * L * S * 4,
        "flops": 2 * B * L * H * D * S,
        "int8_bytes": (B * L * H * D + S * H * D) * 1 + B * L * S * 4,
    }
    return (bf(q), bf(k), bf(w), meta)


def reference(q, k, w, params=None, chunk=None):
    """fp32 参考。

    logits[i,j] = sum_h relu(q[i,h,:] . k[j,:]) * w[i,h]

    **实现要点（都写在 contract 的不变量里）：**
      · relu 在点积之后、乘 W 之前
      · W 是 per-(b,l,h) 的标量
      · fp32 累加

    内存注意：q @ k^T 的中间结果是 [B,L,H,S]，prefill 时会爆内存。
    chunk 参数就是为此 —— 按 L 分块算。
    """
    qf, kf, wf = q.float(), k.float(), w.float()   # [B,L,H,D], [S,H,D], [B,L,H]
    B, L, H, D = qf.shape
    S = kf.shape[0]

    step = int(chunk) if chunk else max(1, L)
    outs = []
    for lo in range(0, L, step):
        hi = min(lo + step, L)
        qb = qf[:, lo:hi]                            # [B,h,H,D]
        # 逐头算，避免 [B,h,H,S] 大中间量同时驻留
        acc = None
        for h in range(H):
            dots = qb[:, :, h, :] @ kf[:, h, :].t()  # [B,h,S]
            relu = dots.clamp_min(0.0)
            term = relu * wf[:, lo:hi, h].unsqueeze(-1)
            acc = term if acc is None else acc + term
        outs.append(acc)
    return torch.cat(outs, dim=1)                    # [B,L,S]


def runner_kwargs(params):
    B, L, H, D, S = _dims(params)
    return {"B": B, "L": L, "H": H, "D": D, "S": S}


def get_runner(name, **kwargs):
    if name == "ref-heavy":
        L = int(kwargs.get("L", 0)) or None
        # prefill 大 L 时按 1024 分块，否则 [B,L,H,S] 中间量会 OOM
        chunk = 1024 if (L and L > 1024) else None

        def _ref(q, k, w, **kw):
            return reference(q, k, w, chunk=chunk)
        return _ref
    raise ValueError("未知变体 %r（可选 %s）" % (name, " / ".join(VARIANTS)))


def run_isolated(fn, *inp):
    copies = [t.clone() if torch.is_tensor(t) else t for t in inp]
    out = fn(*copies)
    return out, []


def compare(a, b):
    af, bf = a.float(), b.float()
    diff = (af - bf).abs()
    denom = af.norm() * bf.norm()
    cos = float((af.flatten() @ bf.flatten()) / denom) if denom > 0 else 1.0
    return {"max_abs": float(diff.max()), "mean_abs": float(diff.mean()), "cosine": cos}


def achieved_bandwidth(median_ms, meta):
    peak_gbs = 1680.0
    gbs = meta["bytes_fused"] / (median_ms * 1e-3) / 1e9
    return {"achieved_gbs": gbs, "peak_gbs": peak_gbs, "ratio": gbs / peak_gbs}
