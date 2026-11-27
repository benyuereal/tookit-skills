"""op-gemm-bf16 的 adapter。

契约见 contract.yaml。inputs 顺序：(A, B) -> out

**这张卡是选题不是优化题**：目标是搞清该用哪个库、上限多少，
不是写一个比 hipBLASLt 快的 GEMM。见 contract.yaml 的 selection_note。
"""
import torch

VARIANTS = ("ref-fp32", "torch-mm", "lightop-gemm", "aiter-gemm")
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
    if name == "aiter-gemm":
        def _ai(A, B, **kw):
            """aiter 的 bf16 GEMM。

            **入参约定**：吃 (A[M,K], B[N,K])，与 lightop.gemm 一样需要转置。
            实测（2026-11-27）：传 B[K,N] 误差 1.36；传 B.T 误差 0.0037 合格。
            """
            import aiter
            return aiter.gemm_a16w16_opus(A, B.t().contiguous())
        return _ai
    if name == "lightop-gemm":
        def _lo(A, B, **kw):
            """lightop.gemm 吃 (A[M,K], B[N,K])，**不是** (A, B[K,N])。

            实测（2026-11-26）：直接传 B[K,N] 误差 1.36（完全错），
            传 B.T 误差 0.0016（合格）。契约规定 A 是 [M,K]、B 是 [K,N]，
            所以这里必须转置 —— 转换放在 adapter 内部，不改变契约。
            """
            import lightop
            # **实测（2026-11-26）：lightop.gemm 在 M >= 24 时结果错误。**
            #   M=8/16, K=512, N=512  -> 误差 0.0022（合格）
            #   M=24,    K=512, N=512 -> 误差 1.0643（完全错）
            #   M=32,    K=512, N=512 -> 误差 4.4751（完全错）
            # 与 K 无关 —— 上一轮先归因到 K>256，是没控制变量的误判。
            # 猜测是它的 tile 分块假设了 M<=16 的 warp 布局，但未查证。
            if A.shape[0] >= 24:
                raise RuntimeError(
                    "lightop.gemm 在 M=%d 时结果不可信（M<=16 正确、M>=24 错），"
                    "见 candidates.yaml 的 known_issue。已拒绝运行，避免污染数据。"
                    % A.shape[0])
            return lightop.gemm(A, B.t().contiguous()).to(torch.bfloat16)
        return _lo
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
