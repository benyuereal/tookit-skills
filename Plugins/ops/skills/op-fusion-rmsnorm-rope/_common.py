import math

import torch

VARIANTS = ("ref-heavy", "torch-unfused")
INPUT_NAMES = ("q", "k", "w_q", "w_k", "cos_sin_cache", "positions")
EPS = 1e-6


def _dims(params):
    return int(params["B"]), int(params["S"])


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
    """构造 (q, k, w_q, w_k, cos_sin_cache, positions, meta)。

    cos_sin_cache 按 [max_pos, 2*D] 存：[0:D] 是 cos，[D:2D] 是 sin。
    这样取一行就是全部需要的三角函数值，不用在 kernel 里现算。
    """
    B, S = _dims(params)
    H = int(params.get("H", 32))
    H_kv = int(params.get("H_kv", 8))
    D = int(params.get("D", 128))
    max_pos = int(params.get("max_pos", 4096))
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    q = torch.randn(B, S, H, D, generator=g, dtype=torch.float32)
    k = torch.randn(B, S, H_kv, D, generator=g, dtype=torch.float32)
    w_q = torch.randn(D, generator=g, dtype=torch.float32) * 0.1 + 1.0
    w_k = torch.randn(D, generator=g, dtype=torch.float32) * 0.1 + 1.0

    t = torch.arange(max_pos, dtype=torch.float32)
    inv = 1.0 / (10000.0 ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
    freqs = torch.outer(t, inv)
    emb = torch.cat([freqs, freqs], dim=-1)
    cache = torch.cat([emb.cos(), emb.sin()], dim=-1)
    pos = torch.randint(0, max_pos, (B, S), generator=g)

    bf = lambda x: x.to(torch.bfloat16).to(dev)
    meta = {
        "B": B, "S": S, "H": H, "H_kv": H_kv, "D": D,
        "shape": params.get("_name", ""),
        "bytes_fused": 2 * (B * S * (H + H_kv) * D * 2),
        "bytes_unfused": 3 * (B * S * (H + H_kv) * D * 2),
    }
    return (bf(q), bf(k), bf(w_q), bf(w_k), cache.to(dev), pos.to(dev), meta)


def _rope(x, cos, sin):
    """interleaved rope：偶数位配 -sin，奇数位配 sin。

    注意 cos/sin 是 [B,S,1,D] 广播到所有头。
    """
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    c = cos[..., 0::2]
    s = sin[..., 0::2]
    o1 = x1 * c - x2 * s
    o2 = x2 * c + x1 * s
    out = torch.empty_like(x)
    out[..., 0::2] = o1
    out[..., 1::2] = o2
    return out


def reference(q, k, w_q, w_k, cos_sin_cache, positions, params=None, chunk=None):
    """fp32 参考：先 rmsnorm（eps 在 sqrt 内）再 rope。

    **刻意不用 torch 的 fused 实现** —— 那是被测对象之一。
    """
    D = q.shape[-1]
    cache = cos_sin_cache.float()
    idx = positions.reshape(-1)
    row = cache.index_select(0, idx).reshape(*positions.shape, 2 * D)
    cos = row[..., :D].unsqueeze(2)
    sin = row[..., D:].unsqueeze(2)

    def _norm_rope(x, w):
        xf = x.float()
        rms = torch.sqrt(xf.pow(2).mean(dim=-1, keepdim=True) + EPS)
        return _rope((xf / rms) * w.float(), cos, sin)

    return _norm_rope(q, w_q), _norm_rope(k, w_k)


def runner_kwargs(params):
    B, S = _dims(params)
    return {"B": B, "S": S}


def get_runner(name, **kwargs):
    if name == "ref-heavy":
        def _ref(q, k, w_q, w_k, cache, pos, **kw):
            return reference(q, k, w_q, w_k, cache, pos)
        return _ref
    if name == "torch-unfused":
        def _unfused(q, k, w_q, w_k, cache, pos, **kw):
            """两趟实现：norm 写出中间态，再读回来做 rope。

            这是**融合要打败的对象**。与 reference 的数学等价，
            区别是把中间态真的写出去再读回来 —— 融合省的正是这一趟。
            """
            D = q.shape[-1]
            cache_f = cache.float()
            idx = pos.reshape(-1)
            row = cache_f.index_select(0, idx).reshape(*pos.shape, 2 * D)
            cos = row[..., :D].unsqueeze(2)
            sin = row[..., D:].unsqueeze(2)

            def _two_pass(x, w):
                # 第一趟：norm，结果写成 bf16 落盘
                xf = x.float()
                rms = torch.sqrt(xf.pow(2).mean(dim=-1, keepdim=True) + EPS)
                mid = ((xf / rms) * w.float()).to(torch.bfloat16)
                # 第二趟：读回来做 rope（这里 bf16 -> float 就是那次往返）
                return _rope(mid.float(), cos, sin)

            return _two_pass(q, w_q), _two_pass(k, w_k)
        return _unfused
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
