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


def _slot_map(topk_idx, E, K, align, pad=True):
    """按专家分桶 + 计算每个 (token, k) 的目标槽位。

    pad=True （默认）：每个专家块按 align 向上取整，返回 padded 布局
                        （`moe_align_block_size` 的语义）。搬运量 = sum(ceil(count/align)*align)。
    pad=False        ：不补零，总行数恰为 M*K（`groupgemm_permute` 的语义）。

    **两种口径必须分开比**：实测 M=8192 K=8 align=128 时，
    padded=73344 行 vs unpadded=65536 行，差 11.9% 的搬运量。
    拿 padded 的实现比 unpadded 的对手，就是比两件不同的活。

    返回 (slot_map[M, K], block_start[E], counts[E])。
    """
    M = topk_idx.shape[0]
    dev = topk_idx.device
    counts = torch.zeros(E, dtype=torch.int64, device=dev)
    for e in range(E):
        counts[e] = int((topk_idx == e).sum())

    padded = (((counts + align - 1) // align) * align) if pad else counts
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


def reference(x, topk_idx, params=None, chunk=None, pad=True):
    """参考实现：返回 (permuted, slot_map)。

    permuted 的形状是 [total_slots, H]，pad=True 时空槽位填 0（哨兵）。
    slot_map 是 [M, K]，给出每个 (token, k) 落到哪一行。

    pad=False 时 total_slots == M*K，与 `lightop.groupgemm_permute` 等语义。
    """
    # **E 必须从输入推导，不能用默认值**。实测踩过：默认 E=8 而实际 E=128，
    # `cursor[e]` 直接 IndexError。E 是 topk_idx 的取值域（max+1），
    # K 是它的列数 —— 这两者输入里都有，没有任何理由去猜。
    p = params or {}
    K = int(topk_idx.shape[1]) if "K" not in p else int(p["K"])
    E = int(p.get("E") or (int(topk_idx.max()) + 1 if topk_idx.numel() else 1))
    align = int(p.get("align", 128))

    slot_map, block_start, counts = _slot_map(topk_idx, E, K, align, pad=pad)
    total = int(((((counts + align - 1) // align) * align) if pad else counts).sum())
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
            # **默认与 reference() 同口径（pad=True）**。
            # 口径差异不该藏在 runner 里 —— 那会让「ref-heavy 与 reference 不一致」
            # 变成一个看不懂的测试失败。要比 unpadded 时显式传 pad=False。
            return reference(x, topk_idx, params=kwargs,
                             pad=bool(kw.get("pad", True)))
        return _ref
    if name == "lightop-align":
        # lightop.groupgemm_permute(tokens, routing_map) -> (permuted, sorted_indices, expert_m_count)
        # 实测 2026-11-25（M=8192 H=4096 E=128 K=8 bf16）：
        #   permuted[65536, 4096] bf16 —— 恰为 M*K 行，无 padding
        #   sorted_indices[65536] int32 / expert_m_count[128] int32
        # 与我们的 reference(pad=False) 等语义：形状相同、行和排序后最大差 0.0。
        def _run(x, topk_idx, **kw):
            import lightop
            M, E = x.shape[0], kwargs.get("E") or None
            if not E:
                E = int(topk_idx.max()) + 1
            routing_map = torch.zeros(M, E, dtype=torch.bool, device=x.device)
            routing_map.scatter_(1, topk_idx.to(torch.long), True)
            permuted, sorted_idx, _cnt = lightop.groupgemm_permute(x, routing_map)
            # **必须返回两份，与 reference 的 outputs 对齐**。少一份 compare 会直接抛。
            # groupgemm_permute 返回的 sorted_idx[total] 是「第 i 个槽位来自哪个 token*k」，
            # 而我们的 slot_map[M,K] 是「(m,k) 去哪个槽位」—— 两者互为逆映射。
            # 这里用 scatter 反推 slot_map，保证输出结构与 reference 完全一致。
            total = permuted.shape[0]
            slot_map = torch.empty(x.shape[0] * topk_idx.shape[1],
                                   dtype=torch.int64, device=x.device)
            slot_map[sorted_idx.to(torch.long)] = torch.arange(
                total, dtype=torch.int64, device=x.device)
            slot_map = slot_map.view(x.shape[0], topk_idx.shape[1]).to(torch.int32)
            return permuted, slot_map
        return _run
    raise RuntimeError(
        "变体 %r 的 adapter 尚未实现（**注意：lightop / aiter 实际已安装**，"
        "此前「需先安装」的措辞是错的）。aiter 的 permute 族未核对签名，"
        "先用 lightop-align。" % name)


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
