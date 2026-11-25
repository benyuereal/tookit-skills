"""op-gemm-bf16 的 adapter。

契约见 contract.yaml。inputs 顺序：(A, B) -> out

**这张卡是选题不是优化题**：目标是搞清该用哪个库、上限多少，
不是写一个比 hipBLASLt 快的 GEMM。见 contract.yaml 的 selection_note。
"""
import torch

VARIANTS = ("ref-fp32", "torch-mm")
INPUT_NAMES = ("A", "B")


def _dims(params):
    return int(params["M"]), int(params["N"]), int(params["K"])


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
    M, N, K = _dims(params)
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    A = torch.randn(M, K, generator=g, dtype=torch.float32) * 0.1
    B = torch.randn(K, N, generator=g, dtype=torch.float32) * 0.1
    meta = {
        "M": M, "N": N, "K": K, "shape": params.get("_name", ""),
        "bytes_fused": (M * K + K * N + M * N) * 2,
        "bytes_unfused": (M * K + K * N + M * N) * 2,
        "flops": 2 * M * N * K,
    }
    return (A.to(torch.bfloat16).to(dev), B.to(torch.bfloat16).to(dev), meta)


def reference(A, B, params=None, chunk=None):
    """fp32 参考。**刻意用 fp32 的 @，不用 bf16 的 torch.mm** ——
    bf16 是受测精度之一，拿它当 oracle 会掩盖误差。"""
    return (A.float() @ B.float()).to(torch.bfloat16)


def runner_kwargs(params):
    M, N, K = _dims(params)
    return {"M": M, "N": N, "K": K}


def get_runner(name, **kwargs):
    if name not in VARIANTS:
        raise ValueError("未知变体 %r（可选 %s）" % (name, " / ".join(VARIANTS)))
    if name == "ref-fp32":
        def _ref(A, B, **kw):
            return reference(A, B)
        return _ref
    if name == "torch-mm":
        def _mm(A, B, **kw):
            return (A @ B).to(torch.bfloat16)
        return _mm
    raise ValueError(name)


def run_isolated(fn, *inp):
    copies = [t.clone() if torch.is_tensor(t) else t for t in inp]
    return fn(*copies), []


def compare(a, b):
    af, bf = a.float(), b.float()
    diff = (af - bf).abs()
    denom = af.norm() * bf.norm()
    cos = float((af.flatten() @ bf.flatten()) / denom) if denom > 0 else 1.0
    return {"max_abs": float(diff.max()), "mean_abs": float(diff.mean()), "cosine": cos}


def achieved_bandwidth(median_ms, meta):
    """GEMM 要看算力不求带宽，但保留接口一致。"""
    peak_tflops = 60.0      # gfx938 bf16 稠密估算，**未标定**
    tflops = meta["flops"] / (median_ms * 1e-3) / 1e12
    return {"achieved_tflops": tflops, "peak_tflops": peak_tflops,
            "ratio": tflops / peak_tflops, "note": "峰值未标定，ratio 仅供参考"}
