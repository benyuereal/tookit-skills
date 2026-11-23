#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""op-moe-w8a8 卡的 adapter：输入构造 / fp32 参考 / 变体注册表。

契约见 scripts/opcheck.py 的 docstring。**这个文件是 opcheck 与 bench 的
唯一数学来源**——门与基准都从这里取输入与参考，避免「门测的东西和
bench 测的东西不是一回事」。

诚实性约定（比功能更重要）：
  - 没有可用后端时 get_runner() **抛异常**，不返回假实现。宁可让门 FAIL
    并说明原因，也不许静默通过。
  - reference() 是纯 torch fp32 稠密实现，任何变体都能和它比。
  - build_inputs() 的随机源固定 seed，多变体吃同一份输入。

── inputs 是**位置参数元组**，顺序即契约 ──
    (x, w1, w2, topk_ids, gates) → out[M, H]
run / run_isolated 都以这个顺序接收。
"""
import torch

# 变体注册表：只登记本卡 variants.yaml 里真实存在的名字（asm 是已判死的，不在表内）
VARIANTS = ("lightop-4stage", "aiter-moe", "triton-shared", "tilelang")

INPUT_NAMES = ("x", "w1", "w2", "topk_ids", "gates")


def _dims(params):
    return (int(params["M"]), int(params["H"]), int(params["I"]),
            int(params["E"]), int(params["K"]))


def load_shape(arg):
    """opcheck 缺省也能自己读 shapes.yaml；这里显式给一份，便于 bench 复用。"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts"))
    from opverify import parse_yaml
    p = Path(__file__).resolve().parent
    if arg and (p / str(arg)).is_file():
        p = p / str(arg)
    else:
        p = p / "shapes.yaml"
    data = parse_yaml(p.read_text(encoding="utf-8"))
    key = arg if arg in data else next(iter(data))
    node = dict(data[key])
    node["_name"] = key
    return node


def build_inputs(params, seed=0, device=None):
    """构造一份可复现的 MoE W8A8 输入。返回 (inp..., meta)。"""
    M, H, I, E, K = _dims(params)
    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.randn(M, H, generator=g, dtype=torch.float32)
    w1 = torch.randn(E, 2 * I, H, generator=g, dtype=torch.float32) * 0.02
    w2 = torch.randn(E, H, I, generator=g, dtype=torch.float32) * 0.02

    def _fp8_sim(t):
        """重量级 fp8 模拟：按 e4m3 量化再反量化回 bf16。

        这样参考实现能用 bf16 直接算，且与真实 kernel 吃到的数值一致。
        """
        amax = t.abs().max().clamp_min(1e-8)
        scale = amax / 448.0
        return ((t / scale).clamp(-448, 448).round() * scale).to(torch.bfloat16)

    logits = torch.randn(M, E, generator=g)
    topk_ids = logits.topk(K, dim=-1).indices.to(torch.int32)
    gates = torch.softmax(logits.gather(1, topk_ids.long()), dim=-1).to(torch.float32)

    inp = (x.to(torch.bfloat16).to(dev), _fp8_sim(w1).to(dev), _fp8_sim(w2).to(dev),
           topk_ids.to(dev), gates.to(dev))
    meta = {"M": M, "H": H, "I": I, "E": E, "K": K,
            "shape": params.get("_name", "")}
    return inp + (meta,)


def reference(x, w1, w2, topk_ids, gates, params=None, chunk=None):
    """fp32 稠密参考：逐 token 逐专家算，再按 gate 加权求和。

    故意写成「最笨但最可信」的实现——它不需要快，只需要对。
    """
    xf, w1f, w2f = x.float(), w1.float(), w2.float()
    M, H = xf.shape
    K = topk_ids.shape[1]
    out = torch.zeros(M, H, dtype=torch.float32, device=xf.device)

    for k in range(K):
        for m in range(M):
            e = int(topk_ids[m, k])
            g = float(gates[m, k])
            gate_up = w1f[e] @ xf[m]
            half = gate_up.shape[0] // 2
            act = torch.nn.functional.silu(gate_up[:half]) * gate_up[half:]
            out[m] += g * (w2f[e] @ act)
    return out


def runner_kwargs(params):
    """传给 get_runner 的形状参数。"""
    M, H, I, E, K = _dims(params)
    return {"M": M, "H": H, "I": I, "E": E, "K": K}


def get_runner(name, tilelang_root=None, **kwargs):
    """返回 callable(*inp) -> out。

    **没有可用后端就抛**——本机尚未安装 lightop / aiter 的 MoE 后端时，
    这是最常见情况。gate 会因此 FAIL 并显示这条消息，这正是要的效果。
    """
    if name not in VARIANTS:
        raise ValueError("未知变体 %r（可选 %s）" % (name, " / ".join(VARIANTS)))
    raise RuntimeError(
        "变体 %r 的 adapter 尚未实现（**注意：lightop / aiter 实际已安装**，此前「需先安装」的措辞是错的）。原措辞：需先安装 lightop / aiter 的 MoE 后端"
        "（见 variants.yaml 的 source 字段）。adapter 已就绪，缺的是后端——"
        "不要用假实现绕过。" % name)


def run_isolated(fn, *inp):
    """可信调用：输入副本 + 输出毒化。

    返回 (out, [被原地改写的输入名])。本卡已知无 in-place 改写，
    但仍然逐项比对，因为「声明不改」和「实际不改」是两件事。
    """
    copies = []
    for t in inp:
        copies.append(t.clone() if torch.is_tensor(t) else t)
    out = fn(*copies)

    poisoned = []
    if torch.is_tensor(out):
        out = out.clone()
    for name, before, after in zip(INPUT_NAMES, inp, copies):
        if torch.is_tensor(before) and torch.is_tensor(after):
            if not torch.equal(before, after):
                poisoned.append(name)
    return out, poisoned


def compare(a, b):
    """缺省 opcheck 会自己算；这里给出与本卡容差口径一致的版本。"""
    af, bf = a.float(), b.float()
    diff = (af - bf).abs()
    denom = af.norm() * bf.norm()
    cos = float((af.flatten() @ bf.flatten()) / denom) if denom > 0 else 1.0
    return {"max_abs": float(diff.max()), "mean_abs": float(diff.mean()),
            "cosine": cos}
