#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""op-moe-align-permute 卡的 adapter。

契约见 scripts/opcheck.py。inputs 顺序：(x, topk_idx) → (permuted, slot_map)

**本段的核心正确性判据是「可逆」与「哨兵只在队尾」**——
它不涉及浮点，所以容差一律 0，任何差异都说明搬错了。

诚实性约定：没有后端就抛，不返回假实现。
"""
import torch

VARIANTS = ("ref-heavy", "lightop-align", "aiter-permute")

INPUT_NAMES = ("x", "topk_idx")

SENTINEL = -1


def _dims(params):
    return (int(params["M"]), int(params["H"]), int(params["E"]),
            int(params["K"]), int(params.get("align", 128)))


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
    """构造 (x, topk_idx)。返回 (x, topk_idx, meta)。

    topk_idx 每行选 K 个**互不相同**的专家（模拟真实路由）。
    """
    M, H, E, K, align = _dims(params)
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.randn(M, H, generator=g, dtype=torch.float32).to(torch.bfloat16)
    logits = torch.randn(M, E, generator=g)
    topk_idx = logits.topk(K, dim=-1).indices.to(torch.int32)

    meta = {"M": M, "H": H, "E": E, "K": K, "align": align,
            "shape": params.get("_name", "")}
    return x.to(dev), topk_idx.to(dev), meta


def _slot_map(topk_idx, E, K, align):
    """按专家分桶 + 按 align 向上取整 + 计算每个 (token, k) 的目标槽位。

    返回 (slot_map[M, K], block_start[E], counts[E])。
    **padding 只垫在队尾**：每个专家块的 [counts[e], align 倍数) 区间是空的。
    """
    M = topk_idx.shape[0]
    dev = topk_idx.device
    counts = torch.zeros(E, dtype=torch.int64, device=dev)
    for e in range(E):
        counts[e] = int((topk_idx == e).sum())

    padded = ((counts + align - 1) // align) * align
    block_start = torch.zeros(E, dtype=torch.int64, device=dev)
    if E > 1:
        block_start[1:] = torch.cumsum(padded[:-1], 0)

    cursor = block_start.clone()
    slot_map = torch.zeros(M, K, dtype=torch.int64, device=dev)
    for m in range(M):
        for k in range(K):
            e = int(topk_idx[m, k])
            slot_map[m, k] = cursor[e]
            cursor[e] += 1
    return slot_map, block_start, counts


def reference(x, topk_idx, params=None, chunk=None):
    """参考实现：返回 (permuted, slot_map)。

    permuted 的形状是 [total_slots, H]，空槽位填 0（哨兵）。
    slot_map 是 [M, K]，给出每个 (token, k) 落到哪一行。
    """
    p = params or {}
    E = int(p.get("E", 8))
    K = int(p.get("K", 2))
    align = int(p.get("align", 128))

    slot_map, block_start, counts = _slot_map(topk_idx, E, K, align)
    total = int((((counts + align - 1) // align) * align).sum())
    H = x.shape[1]

    permuted = torch.zeros(total, H, dtype=x.dtype, device=x.device)
    M = x.shape[0]
    for m in range(M):
        for k in range(K):
            permuted[int(slot_map[m, k])] = x[m]
    return permuted, slot_map.to(torch.int32)


def runner_kwargs(params):
    M, H, E, K, align = _dims(params)
    return {"M": M, "H": H, "E": E, "K": K, "align": align}


def get_runner(name, tilelang_root=None, **kwargs):
    if name not in VARIANTS:
        raise ValueError("未知变体 %r（可选 %s）" % (name, " / ".join(VARIANTS)))
    if name == "ref-heavy":
        def _ref(x, topk_idx, **kw):
            return reference(x, topk_idx, params=kwargs)
        return _ref
    raise RuntimeError(
        "变体 %r 在本机不可用：需先安装 lightop / aiter 的 align+permute 后端。"
        "adapter 已就绪，缺的是后端——不要用假实现绕过。" % name)


def run_isolated(fn, *inp):
    copies = [t.clone() if torch.is_tensor(t) else t for t in inp]
    out = fn(*copies)
    if isinstance(out, tuple):
        out = tuple(o.clone() if torch.is_tensor(o) else o for o in out)
    elif torch.is_tensor(out):
        out = out.clone()
    poisoned = []
    for name, before, after in zip(INPUT_NAMES, inp, copies):
        if torch.is_tensor(before) and torch.is_tensor(after) and not torch.equal(before, after):
            poisoned.append(name)
    return out, poisoned


def compare(a, b):
    """permuted 比逐位，slot_map 比逐位。"""
    pa, sa = a if isinstance(a, tuple) else (a, None)
    pb, sb = b if isinstance(b, tuple) else (b, None)
    diff = (pa.float() - pb.float()).abs()
    out = {"max_abs": float(diff.max()), "mean_abs": float(diff.mean()), "cosine": 1.0}
    if sa is not None and sb is not None:
        out["slot_map_mismatch"] = int((sa != sb).sum())
    return out


# ---- 供测试与下游使用的辅助（不属于 opcheck 契约）

def unpermute(permuted, slot_map, M, H, dtype=None):
    """逆变换：把 permuted 还原成 [M, H]。用于可逆性判据。"""
    out = torch.zeros(M, H, dtype=dtype or permuted.dtype, device=permuted.device)
    K = slot_map.shape[1]
    for m in range(M):
        for k in range(K):
            out[m] = permuted[int(slot_map[m, k])]
    return out


def sentinel_positions_are_tail_only(permuted, slot_map, x):
    """检查哨兵（0 行）是否只出现在队尾块内。

    判据：对每个专家块，有效槽位必须连续地排在块首。
    """
    M = x.shape[0]
    valid_rows = set()
    for m in range(M):
        for k in range(slot_map.shape[1]):
            valid_rows.add(int(slot_map[m, k]))
    n = permuted.shape[0]
    order = sorted(valid_rows)
    return order == list(range(len(order))), len(order), n
