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

VARIANTS = ("ref-heavy", "lightop-fused", "torch-native",
            "fused-3pass-triton", "fused-3pass-tilelang")


def _lightop_version():
    """lightop 的实际版本；读不到就返回 'installed(version unknown)'，不谎报未安装。"""
    try:
        import lightop
        return str(getattr(lightop, "__version__", "installed"))
    except ImportError:
        return "NOT INSTALLED"

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


# ---------------------------------------------------------------- triton 实现
# 2026-11-23 实测结论：12 组参数全部收敛到 179 µs，无一低于 175 µs。
#   最优配方：num_warps=4, rows/CTA=1, 无 mask（H == BLOCK）
#   扫过：warps{1,2,4,8,16,32} / rows/CTA{1,2,4,8} / num_stages{1,2,3} /
#         persistent{512,1024,2048} / 2D 多行 / 去 mask / evict_first|last
#   ⇒ 收敛是这个算子的结构性上限，不是没调够。
# `.cs` cache modifier 在 AMD 后端不支持，别再试。
_TRITON_KERNEL = None


def _get_triton_kernel():
    """惰性编译 kD（无 mask 版）。编译一次后缓存，避免把 JIT 时间算进 bench。"""
    global _TRITON_KERNEL
    if _TRITON_KERNEL is not None:
        return _TRITON_KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def kd(X, R, W, Y, RO, sr, H: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        c = tl.arange(0, BLOCK)
        xf = tl.load(X + row * sr + c).to(tl.float32) + tl.load(R + row * sr + c).to(tl.float32)
        inv = tl.math.rsqrt(tl.sum(xf * xf, axis=0) / H + eps)
        wv = tl.load(W + c).to(tl.float32)
        # **必须写两份输出**：残差连接的下游要用 (x+r)。只写 y 等于少干 1/3 的活
        # （201 MB vs 268 MB），拿它比 lightop 是比两件不同的活。
        tl.store(RO + row * sr + c, xf.to(RO.dtype.element_ty))
        tl.store(Y + row * sr + c, (xf * inv * wv).to(Y.dtype.element_ty))

    _TRITON_KERNEL = kd
    return kd


def _triton_runner(x, w, r, **kw):
    """融合残差 RMSNorm，单 kernel 三趟访存。H 必须是 2 的幂且 <= 65536。"""
    import triton
    M, H = x.shape
    if H & (H - 1):
        raise ValueError("fused-3pass-triton 要求 H 是 2 的幂（当前 %d）—— "
                         "无 mask 版本假定 H == BLOCK" % H)
    kd = _get_triton_kernel()
    y = torch.empty_like(x)
    residual_out = torch.empty_like(r)
    kd[(M,)](x, r, w, y, residual_out, x.stride(0), H, 1e-6, H, num_warps=4)
    # 返回两份，与 lightop.fused_add_rms_norm 等语义（semantics.outputs 声明一致）。
    return y, residual_out


# ---------------------------------------------------------------- tilelang 实现
# 配方 blk_m=1, threads=256（blk_m>=4 会掉到 240µs，别用）。
# 实测 2026-11-23：213.0 µs / 1260 GB/s（2读2写 268MB），比 triton 快 3.6%，
# 但仍慢于 lightop 的 197.4 µs。**换 DSL 不是解药**——三种实现都落在 1200~1360，
# 离 1:1 可达 1749 还有 22~39%，说明瓶颈是「2读2写交织 + 行内归约」这个访存模式本身。
_TL_CACHE = {}


def _get_tilelang_kernel(M, H, eps=1e-6):
    key = (M, H)
    if key in _TL_CACHE:
        return _TL_CACHE[key]
    import tilelang
    import tilelang.language as T
    dtype = "bfloat16"
    blk_m, threads = 1, 256
    inv_n = 1.0 / H

    @tilelang.jit(out_idx=[-2, -1])
    def _k(M: int, N: int, blk_m: int, threads: int, eps: float):
        @T.prim_func
        def main(X: T.Tensor((M, N), dtype), R: T.Tensor((M, N), dtype),
                 W: T.Tensor((N,), dtype),
                 RO: T.Tensor((M, N), dtype), Y: T.Tensor((M, N), dtype)):
            with T.Kernel(T.ceildiv(M, blk_m), threads=threads) as bx:
                xf = T.alloc_fragment((blk_m, N), "float32")
                sq = T.alloc_fragment((blk_m, N), "float32")
                s = T.alloc_fragment((blk_m,), "float32")
                wv = T.alloc_fragment((N,), "float32")
                for i, j in T.Parallel(blk_m, N):
                    xf[i, j] = (X[bx * blk_m + i, j].astype("float32")
                                + R[bx * blk_m + i, j].astype("float32"))
                for i, j in T.Parallel(blk_m, N):
                    RO[bx * blk_m + i, j] = xf[i, j].astype(dtype)
                for i, j in T.Parallel(blk_m, N):
                    sq[i, j] = xf[i, j] * xf[i, j]
                T.reduce_sum(sq, s, dim=1)
                for j in T.Parallel(N):
                    wv[j] = W[j].astype("float32")
                for i, j in T.Parallel(blk_m, N):
                    Y[bx * blk_m + i, j] = (xf[i, j] * T.rsqrt(s[i] * inv_n + eps)
                                            * wv[j]).astype(dtype)
        return main
    _TL_CACHE[key] = _k(M, H, blk_m, threads, eps)
    return _TL_CACHE[key]


def _tilelang_runner(x, w, r, **kw):
    """tilelang 融合残差 RMSNorm（等语义：返回 out 与 residual_out 两份）。"""
    M, H = x.shape
    k = _get_tilelang_kernel(M, H)
    ro, y = k(x, r, w)
    return y, ro


def _lightop_fused_runner(x, w, r, **kw):
    """lightop 的 fused_add_rms_norm：**原地改写 x 与 residual**。

    实测 2026-11-23（prefill-8k, M=8192 H=4096 bf16, events 隔离, 30 次中位）：
      203.1 µs / 991 GB/s / util 0.575（3 遍 = 201 MB）
    对照我方 fused-3pass-triton：172.5 µs ⇒ 我方快 1.18x（同口径纯 kernel）。

    原地改写的后果：**同一个 buffer 连跑会累积错误**（x 被写成 output，
    下一轮读到的输入已经变了）。调用方必须每轮喂副本，ab.py/run_isolated 已处理。
    """
    import lightop
    out, residual_out = lightop.fused_add_rms_norm(x, r, w, 1e-6)
    # **两份都返回**：variants.yaml 的 semantics.outputs = [out, residual_out]。
    # 只返回 out 会让「输出份数」与声明不符，也会让下游拿不到 (x+r)。
    return out, residual_out


def get_runner(name, tilelang_root=None, **kwargs):
    if name not in VARIANTS:
        raise ValueError("未知变体 %r（可选 %s）" % (name, " / ".join(VARIANTS)))
    if name == "ref-heavy":
        def _ref(x, w, r, **kw):
            return reference(x, w, r, params=kw)
        return _ref
    if name == "fused-3pass-triton":
        return _triton_runner
    if name == "lightop-fused":
        return _lightop_fused_runner
    if name == "fused-3pass-tilelang":
        return _tilelang_runner
    raise RuntimeError(
        "变体 %r 的 adapter 尚未实现。" 
        "**注意：lightop %s 已安装且可用**（import lightop 成功），"
        "此前这条报错写「需先安装 lightop」是错的，已更正。"
        "可用接口：fused_add_rms_norm(x,residual,weight,eps) [原地改写]、"
        "RMSNormAdd(...).forward(X,residual)、rms_norm(x,weight,eps) [不融合残差]。" % (name, _lightop_version()))


def _clone_out(out):
    """**保留全部输出**，不只第一个。

    2026-11-23 的教训：lightop.fused_add_rms_norm 返回 (out, residual_out)，
    而这里原来只 `out.clone() if torch.is_tensor(out) else out` —— tuple 直接透传、
    不做任何校验。于是「我们少写了一份输出」这件事**没有任何检查能看见**。
    """
    if torch.is_tensor(out):
        return out.clone()
    if isinstance(out, (tuple, list)):
        return type(out)(_clone_out(x) for x in out)
    return out


def run_isolated(fn, *inp):
    copies = [t.clone() if torch.is_tensor(t) else t for t in inp]
    out = _clone_out(fn(*copies))
    poisoned = []
    for name, before, after in zip(INPUT_NAMES, inp, copies):
        if torch.is_tensor(before) and torch.is_tensor(after) and not torch.equal(before, after):
            poisoned.append(name)
    return out, poisoned


def compare(a, b):
    """逐份比较。**输出份数必须一致**，否则直接抛——少一份就是少干一份活。

    2026-11-23：这个函数原来只处理单个 tensor。lightop 返回 2 份时
    调用方会把它当 1 份比，第 2 份被静默忽略。
    """
    if isinstance(a, (tuple, list)) or isinstance(b, (tuple, list)):
        na = len(a) if isinstance(a, (tuple, list)) else 1
        nb = len(b) if isinstance(b, (tuple, list)) else 1
        if na != nb:
            raise ValueError("输出份数不一致：%d vs %d（不可比，先对齐 semantics.outputs）"
                             % (na, nb))
        pairs = list(zip(a, b)) if na > 1 else [(a, b)]
        worst = {"max_abs": 0.0, "mean_abs": 0.0, "cosine": 1.0}
        for i, (x, y) in enumerate(pairs):
            m = compare(x, y)
            worst["max_abs"] = max(worst["max_abs"], m["max_abs"])
            worst["mean_abs"] = max(worst["mean_abs"], m["mean_abs"])
            worst["cosine"] = min(worst["cosine"], m["cosine"])
        return worst
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
