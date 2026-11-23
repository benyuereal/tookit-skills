#!/usr/bin/env python3
"""op-sparse-mla 的公共装配：形状加载、输入构造、参考实现、变体适配。

bench/ 与 tests/ 都从这里取，避免各写一套口径。

典型用法：
    from _common import load_shape, build_inputs, ref_fp32_chunked, get_runner
"""

import importlib.util
import math
import os
import sys
from pathlib import Path

CARD_DIR = Path(__file__).resolve().parent
# 默认形状 = DeepSeek DSA / V4.1 Flash：q/kv 576 宽（512 latent + 64 rope tail），out 512 宽。
# GLM5.3 flash 是 d_qk == d_v == 512（tail_dim = 0），由档位 yaml 的 d_v / tail_dim 覆盖。
DQK = 576
DV = 512
TAIL = DQK - DV
SM_SCALE = 1.0 / math.sqrt(DQK + TAIL)   # MLA 惯例：1/sqrt(576+64)


def dims_from_params(params):
    """从档位参数取 (d_qk, d_v, tail_dim)。

    缺省 = 576 / 512 / 64（DeepSeek DSA）。GLM5.3 flash 写 `d_v: 512, tail_dim: 0`，
    即 q/kv 宽 512、无 rope 尾、out 宽 512。
    """
    d_v = int(params.get("d_v", DV))
    tail = int(params.get("tail_dim", TAIL))
    d_qk = int(params.get("d_qk", d_v + tail))
    return d_qk, d_v, tail


def sm_scale_for(params):
    """MLA 惯例 sm_scale = 1/sqrt(d_qk + tail_dim) = 1/sqrt(d_v + 2*tail_dim)。

    576/64 -> 1/sqrt(640)（= 常量 SM_SCALE）；512/0 -> 1/sqrt(512)。
    """
    d_qk, d_v, tail = dims_from_params(params)
    return 1.0 / math.sqrt(d_qk + tail)


# ---------------------------------------------------------------- 共享层定位

def find_scripts_dir():
    """定位仓库/shared 层的 scripts 目录（opbench.py / opverify.py 所在处）。"""
    cands = []
    env = os.environ.get("DCU_TOOLKIT_SHARED")
    if env:
        cands.append(Path(env) / "scripts")
    cands.append(CARD_DIR.parents[3] / "scripts")          # <repo>/Plugins/ops/skills/<card> -> <repo>/scripts
    cands.append(CARD_DIR.parent / "_dcu-toolkit-shared" / "scripts")  # 安装态
    for c in cands:
        if (c / "opbench.py").exists():
            return c
    raise SystemExit(
        "找不到 scripts/opbench.py。请设置 DCU_TOOLKIT_SHARED 指向共享层根目录，"
        "或从仓库内运行（<repo>/Plugins/ops/skills/op-sparse-mla）。候选：%s"
        % [str(c) for c in cands]
    )


sys.path.insert(0, str(find_scripts_dir()))
from opbench import bench_fn, significant, floor_status, emit_metric, dump_json  # noqa: E402
from opverify import parse_yaml  # noqa: E402


# ---------------------------------------------------------------- 形状

def load_shape(arg, shapes_dir=None):
    """arg = 档位名（`decode-s128`），也可以是 yaml 路径。

    所有档位在同一张 `shapes.yaml` 里（一档一个顶层键）：原先一档一文件时，7 个档位
    共用同一份 schema，改一处字段要改 7 个文件、漂了也看不出来。
    显式路径若指向**单档文件**（顶层有 params）也照收 —— 复现脚本可以冻结一份当时的口径。
    """
    p = Path(arg)
    if p.suffix in (".yaml", ".yml"):
        if not p.exists():
            raise SystemExit("找不到形状文件: %s" % p)
        with open(p, "r", encoding="utf-8") as f:
            raw = parse_yaml(f.read())
        if "params" in raw:                     # 冻结下来的单档文件
            raw.setdefault("shape", p.stem)
            raw["_path"] = str(p)
            return raw
        table_path, table = p, raw
    else:
        table_path = (Path(shapes_dir) if shapes_dir else CARD_DIR) / "shapes.yaml"
        with open(table_path, "r", encoding="utf-8") as f:
            table = parse_yaml(f.read())
    if arg not in table:
        raise SystemExit("形状档位 %r 不在 %s（有：%s）"
                         % (arg, table_path, ", ".join(sorted(table)) or "无"))
    cfg = dict(table[arg])
    cfg["shape"] = arg
    cfg["_path"] = "%s#%s" % (table_path, arg)
    return cfg


# ---------------------------------------------------------------- 输入

def build_inputs(params, seed=0, device=None):
    """按档位参数构造 q/kv/indices。返回 (q, kv, indices, meta)。

    两种索引池口径，**由 `pool` 参数选**，默认沿用厂商惯例：

    `pool=window`（默认，对齐厂商自带测试 tilelang_sparse_fwd 的
    test_sparse_mla_fwd / benchmark_glm51）：
      - 第 t 行只从 `[0, t+q_start)` 里选，取 randperm 的前 topk 个（**乱序**，不排序）；
      - caveat **这不是生产口径**：S=32 时 working set 只有 ~4.8 MB、S=4096 也只有 ~9.4 MB，
        全在 L2 里；而生产 decode（ctx=320k）是从整个 377 MB 池随机取。
        只把池从 377 MB 缩到 2.4 MB，flash_mla 快 1.83×、tilelang 只快 1.03×
        ⇒ 用这个口径测出的跨实现比值**系统性低估 tilelang**（fm 是内存受限的一方）。
        判 GEMM 效率用它（热工作集下才有 135 TF）；**跨实现比站位必须用 full**。

    `pool=full`（生产口径，真随机池）：
      - 每行都从 `[0, SKV)` 全池随机取 topk 个 ⇒ working set ≈ SKV×d×2 字节。
      - 跨实现站位、任何「我们领先/落后 x 倍」的结论都必须用这个口径。

    `pad_position=head` 只用于探测边界（tilelang 变体在这种布局下结果错误，见 ledger.md）。
    """
    import torch

    S = int(params.get("S", 128))
    H = int(params.get("H", 16))
    SKV = int(params.get("SKV", 65536))
    G = int(params.get("kv_group", 1))
    topk = int(params.get("topk", 2048))
    q_start = int(params.get("q_start_s_index", 4096))
    pad_pos = str(params.get("pad_position", "tail"))
    eff_ratio = float(params.get("effective_ratio", 1.0))
    pool = str(params.get("pool", "window"))
    if pad_pos not in ("tail", "head"):
        raise SystemExit("pad_position 只能是 tail 或 head，收到 %r" % pad_pos)
    if pool not in ("window", "full"):
        raise SystemExit("pool 只能是 window 或 full，收到 %r" % pool)
    if pool == "full" and SKV < topk:
        # 全池随机取 topk，池比 topk 还小时取不满 ⇒ 与 window 池没有区别，直接拒跑而非静默降级
        raise SystemExit(
            "pool=full 要求 SKV(%d) >= topk(%d)，否则无法从全池取满" % (SKV, topk))
    torch.manual_seed(seed)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    d_qk, d_v, tail = dims_from_params(params)
    q = torch.randn((S, H, d_qk), dtype=torch.bfloat16, device=dev)
    kv = torch.randn((SKV, G, d_qk), dtype=torch.bfloat16, device=dev)
    idx = torch.full((S, G, topk), -1, dtype=torch.int32, device=dev)
    for t in range(S):
        n = SKV if pool == "full" else min(max(1, t + q_start), SKV)
        k = min(topk, n, max(1, int(round(topk * eff_ratio))))
        pick = torch.randperm(n, device=dev)[:k]
        for g in range(G):
            if pad_pos == "head":
                idx[t, g, topk - k:] = pick
            else:
                idx[t, g, :k] = pick
    eff = float((idx >= 0).float().mean().item())
    # 池大小 = 索引**可能**指向的范围，才是决定会不会被 L2 白送的那个量。
    # 不要用 S*topk*d*2 当它：decode-m1 的 q_start = SKV-S（窗口铺满整个 320k 池），
    # 名义读量只有 2.25 MB，但池是 360 MB —— 拿读量当工作集会得出相反的结论。
    pool_extent = SKV if pool == "full" else min(SKV, S + q_start)
    pool_mb = pool_extent * d_qk * 2 / 2 ** 20
    touched_mb = S * topk * d_qk * 2 / 2 ** 20
    meta = {"S": S, "H": H, "SKV": SKV, "kv_group": G, "topk": topk,
            "d_qk": d_qk, "d_v": d_v, "tail_dim": tail,
            "q_start_s_index": q_start, "pad_position": pad_pos,
            "pool": pool, "pool_extent": pool_extent, "pool_mb": pool_mb,
            "touched_mb": touched_mb,
            "effective_ratio": eff, "device": str(dev)}
    # 门槛 64 MB：本机 L2 远小于此，池只要到这个量级就不可能整份驻留。
    # 判据是**池大小**而不是 `pool` 名字 —— window 分支在 q_start 铺满全池时（decode-m1/m32、
    # prefill 两档）与 full 等价，按名字报警会变成狼来了。
    if pool_mb < 64:
        meta["warning"] = (
            "池只有 %.1f MB（索引只可能指向这么多），几乎必然整份驻留 L2 ⇒ 跨实现比值"
            "系统性低估 tilelang；比站位请把 pool 改 full 或加大 q_start" % pool_mb)
    return q, kv, idx, meta


# ---------------------------------------------------------------- 参考

def ref_fp32_chunked(q, kv, idx, sm_scale=None, chunk=4, d_v=None):
    """分块 fp32 参考。一次性构造 [S, SKV] 分数矩阵会 OOM，必须分块。

    宽度从张量本身取（d_qk = q.shape[-1]），输出宽度取 d_v（缺省 = 常量 DV）。
    因此 576/512 与 GLM5.3 的 512/512 共用这一份，不需要两份参考。
    """
    import torch

    d_qk = int(q.shape[-1])
    dv = int(d_v) if d_v is not None else min(DV, d_qk)
    # MLA 惯例 1/sqrt(d_qk + tail_dim)，tail_dim = d_qk - d_v：576/512 -> 1/sqrt(640)，512/512 -> 1/sqrt(512)
    sm_scale = (1.0 / math.sqrt(d_qk + (d_qk - dv))) if sm_scale is None else sm_scale
    S, H, _ = q.shape
    SKV, G, _ = kv.shape
    assert G == 1, "参考实现仅支持 kv_group=1"
    out = torch.empty((S, H, dv), dtype=torch.float32, device=q.device)
    kf = kv[:, 0, :].float()                                   # [SKV, d_qk]
    pad = torch.zeros((1, d_qk), dtype=torch.float32, device=q.device)
    kf_pad = torch.cat([kf, pad], dim=0)                       # 末行作 -1 的落点
    for s0 in range(0, S, chunk):
        s1 = min(S, s0 + chunk)
        ii = idx[s0:s1, 0, :].long()                           # [n, topk]
        valid = ii >= 0
        ii = torch.where(valid, ii, torch.full_like(ii, SKV))
        qf = q[s0:s1].float()                                  # [n, H, 576]
        score = torch.einsum("nhd,ntd->nht", qf, kf_pad[ii]) * sm_scale
        score = score.masked_fill(~valid.unsqueeze(1), float("-inf"))
        score = torch.nan_to_num(score, neginf=-1e30)
        prob = torch.softmax(score, dim=-1)
        out[s0:s1] = torch.einsum("nht,ntd->nhd", prob, kf_pad[ii][..., :dv])
    return out


def compare(a, b):
    """a/b 均为 [S,H,512]；返回误差指标（以 b 为参考）。"""
    import torch

    af = a.reshape(-1, a.shape[-1]).float()
    bf = b.reshape(-1, b.shape[-1]).float()
    d = (af - bf).abs()
    cos = torch.nn.functional.cosine_similarity(af.flatten(), bf.flatten(), dim=0).item()
    return {"max_abs": float(d.max().item()), "mean_abs": float(d.mean().item()), "cosine": float(cos)}


# ---------------------------------------------------------------- 变体

def _load_tilelang_module(root):
    p = Path(root) / "perf" / "sparse_mla" / "sparse_mla_fwd.py"
    if not p.exists():
        raise SystemExit("tilelang 变体路径不存在: %s（用 --tilelang-root 指定）" % p)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("op_sparse_mla_tilelang", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_PREFETCH_CACHE = {}


def _load_prefetch_module(root):
    """把 `bench/prefetch.py` 的改写结果落到临时目录再当模块导入（同 root 只生成一次）。"""
    root = str(Path(root).resolve())
    if root in _PREFETCH_CACHE:
        return _PREFETCH_CACHE[root]
    import tempfile

    # 按路径加载 bench/prefetch.py（bench/ 不是包，避免依赖 sys.path 上的 CARD_DIR）
    _pp = Path(__file__).resolve().parent / "bench" / "prefetch.py"
    _spec = importlib.util.spec_from_file_location("op_sparse_mla_prefetch_gen", _pp)
    _gen = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_gen)
    transform = _gen.transform

    src_path = Path(root) / "perf" / "sparse_mla" / "sparse_mla_fwd.py"
    if not src_path.exists():
        raise SystemExit("tilelang 变体路径不存在: %s（用 --tilelang-root 指定）" % src_path)
    new_src, n = transform(src_path.read_text(encoding="utf-8"))
    if not n:
        raise SystemExit("预取改写没匹配到任何主循环，厂商文件结构可能已变")
    if root not in sys.path:
        sys.path.insert(0, root)
    tmp = Path(tempfile.mkdtemp(prefix="sparse_mla_prefetch_")) / "sparse_mla_prefetch_fwd.py"
    tmp.write_text(new_src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("op_sparse_mla_tilelang_prefetch", tmp)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    _PREFETCH_CACHE[root] = m
    return m


def poison_output(shape, dtype, device):
    """把一块同尺寸显存填成 NaN 再释放，逼后续 torch.empty 复用到它。

    为什么必须做：tilelang/flash_mla 的 wrapper 都用 torch.empty 建输出，**kernel 没写到的
    位置会保留上一次的旧值**。我们踩过：同一个 head-layout 输入两次跑出不同结论，
    一次是真结果、一次是上一次的 tail 结果，vs 参考的误差看起来只有 2e-3。
    毒化后若出现 NaN 就说明该位置没被写。
    """
    import torch

    z = torch.full(tuple(shape), float("nan"),
                   dtype=dtype if dtype.is_floating_point else torch.float32, device=device)
    del z
    torch.cuda.synchronize() if torch.cuda.is_available() else None


def run_isolated(run, q, kv, idx, poison=True, d_v=None):
    """可信调用：毒化输出缓冲 + 给变体独立输入副本，返回 (out, 被原地改写的输入名列表)。

    两份防护缺一不可，都是踩出来的：
      - 输出：wrapper 用 torch.empty 建输出，kernel 没写到的位置保留旧值，
        漏写看起来像「对了」（见 poison_output）。
      - 输入：flash_mla.flash_mla_sparse_fwd 会把 indices 规范化后**原地写回**
        （有效索引挪到前部、-1 挪到队尾，实测幂等、q/kv 不动）。多个变体共用同一份
        indices 时，先跑的那个会改掉后跑那个的输入 —— 曾把「tilelang 前置 -1 输出恒 0」
        的失败掩盖成「两种布局都一致」。
    """
    import torch

    qs, kvs, ixs = q.clone(), kv.clone(), idx.clone()
    if poison:
        poison_output((q.shape[0], q.shape[1], int(d_v) if d_v is not None else DV), q.dtype, q.device)
    out = run(qs, kvs, ixs)
    mutated = []
    if not torch.equal(qs, q):
        mutated.append("q")
    if not torch.equal(kvs, kv):
        mutated.append("kv")
    if not torch.equal(ixs, idx):
        mutated.append("indices")
    return out, mutated


# ---------------------------------------------------------------- 执行身份（T10/P0-1）
#
# 「这个数是用哪份产物跑出来的」必须当场抓，不能事后猜：tilelang 缓存下有 1700+ 个
# executable.so，且生成的代码里没有算子名，事后按名字 grep 一个也匹配不上（已实测）。
_LOADED_MODULES = {}


def so_snapshot():
    """/proc/self/maps 里当前已映射的 .so 绝对路径。取不到返回 None（不是空集）。"""
    import os

    try:
        paths = set()
        with open("/proc/self/maps", "r") as f:
            for line in f:
                parts = line.split(None, 5)
                if len(parts) >= 6 and parts[5].startswith("/"):
                    p = parts[5].strip()
                    if ".so" in p:
                        paths.add(p)
        return paths
    except OSError:
        return None


def _kernel_libpath(kernel):
    """从一个 tilelang JITKernel 上取出它实际加载的 .so（取不到返回 None）。"""
    import os

    for holder in (kernel, getattr(kernel, "adapter", None)):
        if holder is None:
            continue
        for attr in ("libpath", "lib_path", "so_path"):
            v = getattr(holder, attr, None)
            if isinstance(v, str) and v and os.path.isfile(v):
                return v
    return None


def runtime_artifacts(name, maps_before=None):
    """该变体「运行时实际加载」的产物。拿不到就是空列表 —— **不猜路径**。

    优先读 tilelang JITImpl 的 `_kernel_cache`（每个 kernel 的 adapter.libpath 就是
    它真正 dlopen 的那份 .so）；读不到再退回 /proc/self/maps 的增量（只取编译缓存
    目录下的，避免把 torch/tvm 的公共库也记成「算子产物」）。
    """
    import os

    m = _LOADED_MODULES.get(name)
    src = getattr(m, "__file__", None) if m is not None else None
    libs = []
    if m is not None:
        for attr in dir(m):
            obj = getattr(m, attr, None)
            cache = getattr(obj, "_kernel_cache", None)
            if isinstance(cache, dict):
                for k in cache.values():
                    p = _kernel_libpath(k)
                    if p:
                        libs.append(p)
    out = {"source": [src] if src else [], "loaded": sorted(set(libs))}
    if out["loaded"]:
        out["loaded_from"] = "tilelang JITImpl._kernel_cache[].adapter.libpath"
        return out
    if maps_before is not None:
        after = so_snapshot()
        if after is not None:
            cache_dirs = [d for d in (os.environ.get("TILELANG_CACHE_DIR"),
                                      os.path.expanduser("~/.tilelang")) if d]
            new = [p for p in (after - maps_before)
                   if any(p.startswith(d) for d in cache_dirs) and p.endswith(".so")]
            if new:
                out["loaded"] = sorted(new)
                out["loaded_from"] = "/proc/self/maps 增量（限于 tilelang 缓存目录）"
                return out
    out["loaded_note"] = "运行时没取到加载产物路径；不许事后猜，manifest 会因缺 role=loaded 被拒"
    return out


def get_runner(name, tilelang_root=None, d_v=None, sm_scale=None):
    """返回 fn(q, kv, idx) -> out[S,H,d_v]。失败时抛异常，由调用方记 skipped。

    `d_v` / `sm_scale` 缺省 = 常量 DV(512) / SM_SCALE(1/sqrt(640))，即 DeepSeek DSA 口径；
    GLM5.3 flash（d_qk == d_v == 512，无尾）用 `_common.sm_scale_for(params)` + d_v=512 传入。
    """
    dv = int(d_v) if d_v is not None else DV
    scale = SM_SCALE if sm_scale is None else float(sm_scale)

    if name == "tilelang":
        root = tilelang_root or os.environ.get("TILELANG_HYGON_ROOT")
        if not root:
            raise RuntimeError("未指定 tilelang 根目录（--tilelang-root 或 TILELANG_HYGON_ROOT）")
        m = _load_tilelang_module(root)
        _LOADED_MODULES[name] = m

        def run(q, kv, idx):
            out = m.tilelang_sparse_fwd(q, kv, idx, scale, d_v=dv)
            return out.reshape(q.shape[0], q.shape[1], dv)

        return run

    if name == "tilelang-prefetch":
        # 不是仓库里的文件，而是对厂商源码做「下一块 gather 提前」改写后动态生成
        root = tilelang_root or os.environ.get("TILELANG_HYGON_ROOT")
        if not root:
            raise RuntimeError("未指定 tilelang 根目录（--tilelang-root 或 TILELANG_HYGON_ROOT）")
        m = _load_prefetch_module(root)
        _LOADED_MODULES[name] = m

        def run(q, kv, idx):
            out = m.tilelang_sparse_fwd(q, kv, idx, scale, d_v=dv)
            return out.reshape(q.shape[0], q.shape[1], dv)

        return run

    if name == "flash_mla":
        import flash_mla

        def run(q, kv, idx):
            res = flash_mla.flash_mla_sparse_fwd(q, kv, idx, scale, d_v=dv)
            return res[0] if isinstance(res, (tuple, list)) else res

        return run

    if name == "hip-glm":
        # 手写 HIP 前向（dev/sparse_mla_hip_glm.hip，split=2，GLM5.3 吸收式 MLA）。
        # 只覆盖 GLM 档：H=64 / d_qk=512 / d_v=512 / topk=2048 / S 任意（grid 按 S 缩放）。
        import ctypes
        import torch

        so = CARD_DIR / "dev" / "libsparse_mla_hip_glm.so"
        if not so.exists():
            raise RuntimeError("未找到 %s（先 aicc 编译 dev/sparse_mla_hip_glm.hip）" % so)
        lib = ctypes.CDLL(str(so))
        lib.hip_sparse_mla_fwd.restype = None
        lib.hip_sparse_mla_fwd.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p]

        def run(q, kv, idx):
            S, H, d_qk = q.shape
            topk = idx.shape[-1]
            if not (H == 64 and d_qk == 512 and dv == 512 and topk == 2048):
                raise SystemExit(
                    "hip-glm 变体当前只支持 H=64/d_qk=512/d_v=512/topk=2048"
                    "（收到 H=%d d_qk=%d d_v=%d topk=%d）" % (H, d_qk, dv, topk))
            out = torch.empty((S, H, dv), dtype=q.dtype, device=q.device)
            stream = torch.cuda.current_stream().cuda_stream
            lib.hip_sparse_mla_fwd(
                q.data_ptr(), kv.data_ptr(), idx.data_ptr(), out.data_ptr(),
                S, int(kv.shape[0]), ctypes.c_float(scale),
                ctypes.c_void_p(stream))
            return out

        return run

    raise SystemExit("未知变体: %s（可选 tilelang / tilelang-prefetch / flash_mla / hip-glm）" % name)


def tilelang_split_runner(tilelang_root, num_split, num_split_tail=0, d_v=None, tail_dim=None, sm_scale=None):
    """强制指定 num_split 直接调内核，用于 split_invariance。返回 fn(q,kv,idx)->out。

    `d_v` / `tail_dim` 缺省 512 / 64（DeepSeek DSA）；GLM5.3 传 tail_dim=0。
    """
    import torch

    dv = int(d_v) if d_v is not None else DV
    tail = int(tail_dim) if tail_dim is not None else TAIL
    scale = SM_SCALE if sm_scale is None else float(sm_scale)
    m = _load_tilelang_module(tilelang_root)
    cache = {}

    def run(q, kv, idx):
        S, H, _ = q.shape
        topk = idx.shape[-1]
        key = (H, topk)
        if key not in cache:
            kernel = m._get_sparse_mla_fwd_kernel(
                H, dv, tail, topk, int(kv.shape[1]), scale,
                32, 256, 0, num_split, num_split_tail, 1, "bfloat16")
            cache[key] = kernel
        kernel = cache[key]
        out = torch.empty((1, S, H, dv), dtype=q.dtype, device=q.device)
        glse = torch.empty((1, S, num_split, H), dtype=torch.float16, device=q.device)
        partial = torch.empty((1, S, num_split, H, dv), dtype=torch.float16, device=q.device)
        kernel(q.unsqueeze(0), kv.unsqueeze(0), idx.unsqueeze(0), glse, partial, out)
        return out.reshape(S, H, dv)

    return run


# ---------------------------------------------------------------- opcheck adapter 契约
# scripts/opcheck.py 只认这几个符号（模板 ↔ 卡的分界）：流程、报告、退出码归模板，
# 张量构造、oracle、变体注册表归本文件。接口说明见 opcheck.py 的 docstring。

def reference(q, kv, idx, params, chunk=4):
    """高精度 oracle：fp32 分块，d_v 与 sm_scale 都从档位参数推。"""
    _d_qk, d_v, _tail = dims_from_params(params)
    return ref_fp32_chunked(q, kv, idx, sm_scale=sm_scale_for(params), chunk=chunk, d_v=d_v)


def runner_kwargs(params):
    """get_runner 需要的形状参数（宽度换了 sm_scale 就换，别在测试里另算一份）。"""
    _d_qk, d_v, _tail = dims_from_params(params)
    return {"d_v": d_v, "sm_scale": sm_scale_for(params)}


def split_runner(tilelang_root, params, num_split):
    """split_invariance 探针：强制 num_split 直调内核。"""
    _d_qk, d_v, tail = dims_from_params(params)
    return tilelang_split_runner(tilelang_root, num_split=num_split, d_v=d_v,
                                 tail_dim=tail, sm_scale=sm_scale_for(params))


def permute_valid(idx, seed=1234):
    """乱序每一行的有效索引（`-1` 槽位保持不动）。"""
    import torch

    g = torch.Generator(device=idx.device)
    g.manual_seed(seed)
    out = idx.clone()
    S, G, _topk = idx.shape
    for s in range(S):
        for gg in range(G):
            v = idx[s, gg][idx[s, gg] >= 0]
            if v.numel() <= 1:
                continue
            perm = torch.randperm(v.numel(), device=idx.device, generator=g)
            out[s, gg, : v.numel()] = v[perm]
    return out


def head_layout(idx):
    """把每一行的有效索引整体挪到队尾（无效槽位变前缀），有效集合与顺序不变。"""
    import torch

    S, G, topk = idx.shape
    out = torch.full_like(idx, -1)
    for s in range(S):
        for g in range(G):
            v = idx[s, g][idx[s, g] >= 0]
            if v.numel():
                out[s, g, topk - v.numel():] = v
    return out


# ---------------------------------------------------------------- 报告

def make_report(op, variant, shape, stats, meta, extra=None):
    rep = {
        "op": op, "variant": variant, "shape": shape,
        "caliber": stats.get("caliber"),
        "metrics": {
            "median_ms": stats.get("median_ms"),
            "mad_ms": stats.get("mad_ms"),
            "min_ms": stats.get("min_ms"),
            "max_ms": stats.get("max_ms"),
            "repeats": stats.get("n"),
        },
        "params": {k: v for k, v in meta.items() if k != "device"},
    }
    if extra:
        rep.update(extra)
    return rep


def _probe_clock():
    """当场读 sclk/mclk 与 CU 数。

    为什么必须在采集时读、而不是事后补：derive.py 的算力地板 ∝ 1/sclk，
    sclk 在 0.6/1.15 GHz 两档之间切换时算力地板差 1.9×，足以让 cost_bound
    从 memory 翻到 compute。事后探测的时钟解释不了当时测的耗时。

    读不到时返回 unavailable 并写明原因，**不伪装成 unknown 字符串**。
    """
    import subprocess

    out = {}
    try:
        import re
        import subprocess
        r = subprocess.run(["hy-smi", "-c"], capture_output=True, text=True, timeout=5)
        txt = r.stdout or ""
        # 只取 sclk 行：「HCU[0] : sclk clock level: 1 (600Mhz)」
        # 注意不能一把抓所有 MHz —— fclk/mclk/socclk/pcie 都会混进来。
        sclk = [int(m) for m in re.findall(r"sclk clock level:[^(]*\((\d+)\s*Mhz", txt, re.I)]
        mclk = [int(m) for m in re.findall(r"mclk clock level:[^(]*\((\d+)\s*Mhz", txt, re.I)]
        if sclk:
            out["sclk_mhz_per_card"] = sclk
            out["sclk_source"] = "hy-smi -c 逐卡运行档位（未锁频，采集时刻）"
            # 纪律：取**本进程实际用的那张卡**，不取 max（挑好看的那个是禁止的）。
            try:
                import torch
                dev = torch.cuda.current_device() if torch.cuda.is_available() else 0
            except Exception:  # noqa: BLE001
                dev = 0
            out["sclk_device_index"] = dev
            out["sclk_mhz"] = sclk[dev] if dev < len(sclk) else None
            if out["sclk_mhz"] is None:
                out["sclk_unavailable"] = "device %d 超出逐卡列表长度 %d" % (dev, len(sclk))
            if len(set(sclk)) > 1:
                out["sclk_spread"] = {"min": min(sclk), "max": max(sclk),
                                      "note": "卡间档位不一致，跨卡比较只能用比值"}
        if mclk:
            out["mclk_mhz"] = max(mclk)
        if not sclk:
            out["sclk_unavailable"] = "hy-smi -c 输出里没有 sclk 行"
    except Exception as e:  # noqa: BLE001
        out["sclk_unavailable"] = "%s: %s" % (type(e).__name__, e)
    try:
        import torch
        if torch.cuda.is_available():
            out["cu_count"] = torch.cuda.get_device_properties(0).multi_processor_count
    except Exception as e:  # noqa: BLE001
        out["cu_count_unavailable"] = str(e)
    return out


def env_fingerprint():
    import platform

    fp = {"torch": None, "python": platform.python_version(), "device": "cpu"}
    try:
        import torch
        fp["torch"] = torch.__version__
        if torch.cuda.is_available():
            fp["device"] = torch.cuda.get_device_name(0)
            fp["gfx"] = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "unavailable")
    except Exception as e:  # noqa: BLE001
        fp["device_error"] = str(e)
    # dtk 版本只认 `hipconfig --version`：DTK 里没有 /opt/dtk/version 这个文件（实测 2026-10-28），
    # 早先探它导致 dtk 永远是占位符 -> 身份闸永远 FAIL、闸门不可满足。
    # hipconfig 是本仓库 dcu-dtk-layout 里记的官方入口（读 lib/.hipInfo + bin/.hipVersion）。
    for k, cmd in (("dtk", "hipconfig --version 2>/dev/null"),
                   ("rocminfo_gfx", "rocminfo 2>/dev/null | grep -m1 gcnArchName")):
        try:
            import subprocess
            v = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                               timeout=5).stdout.strip()
            fp[k] = v or "unavailable"
        except Exception as e:  # noqa: BLE001
            fp[k] = "unavailable(%s)" % type(e).__name__
    fp.update(_probe_clock())
    return fp
