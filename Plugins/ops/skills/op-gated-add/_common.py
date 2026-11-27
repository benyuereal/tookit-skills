#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""op-gated-add 的 adapter（bootstrap 生成，**需要人工补齐 reference 的数学**）。

生成时间：2026-11-25
生成器：scripts/op_bootstrap.py

## 必须补的地方（否则正确性门跑不了）

1. `reference()` 里的数学 —— 现在只有形状占位，没有真实计算
2. `VARIANTS` —— 至少要有一个真实实现（库调用或手写）
3. `build_inputs()` 的数值分布 —— 要覆盖真实场景的取值范围

## 为什么 reference 不自动生成

参考实现是**正确性的唯一锚**。猜一个参考等于让整个门禁失去意义 ——
它会「通过」一个算错的实现，因为两边都错。
所以这里留占位符 + 让门禁拦住，而不是写一个能跑但不可信的版本。
"""
import torch

# TODO(必填)：填真实实现名
VARIANTS = ("ref-heavy",)

INPUT_NAMES = ("x", "gate",)


def build_inputs(params, seed=0, device=None):
    """造输入。**elementwise 类与数值分布无关**，所以这里可以真的生成。

    reduce / softmax / quant 这些**不行**：分布直接决定命中率和量化误差，
    必须由算子作者按真实场景写（bootstrap 对它们保留 NotImplementedError）。
    """
    import torch
    g = torch.Generator(device="cpu").manual_seed(int(seed))

    def _num(v):
        return int(v) if not isinstance(v, float) else v

    M = _num(params.get("M", 8192))
    H = _num(params.get("H", 4096))
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn((M, H), generator=g, dtype=torch.float32).to(dev).to(torch.bfloat16)
    gate = torch.randn((M, H), generator=g, dtype=torch.float32).to(dev).to(torch.bfloat16)
    return (x, gate)


def reference(x, gate, params=None):
    """fp32 参考。数学：y = x + gate * s

    三条纪律（别省）：
      * 用 fp32 算，最后不转回 bf16 —— 参考是"真值"，转回去等于自己抬高容差；
      * 不 in-place 改输入 —— 否则基准与变体共享同一块内存，测出来的差是脏的；
      * s 从 params 取，取不到用 1.0 并**在报告里可见**（ab.py 会打印 params）。
    """
    import torch
    p = params or {}
    s = float(p.get("s", 1.0))
    xf = x.to(torch.float32)
    gf = gate.to(torch.float32)
    return xf + gf * s


def load_shape(arg=None):
    """读本卡 shapes.yaml。ab.py / loop.py 都靠它拿形状。

    bootstrap 以前不生成这个函数，新卡第一次 `loop.py run` 就报
    `module '_common' has no attribute 'load_shape'`（实测 2026-11-27）。
    """
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[3] / "scripts"))
    from opverify import parse_yaml

    p = _P(__file__).resolve().parent
    target = (p / str(arg)) if (arg and (p / str(arg)).is_file()) else (p / "shapes.yaml")
    data = parse_yaml(target.read_text(encoding="utf-8"))
    key = arg if arg in data else next(iter(data))
    node = dict(data[key])
    node.setdefault("_name", key)
    return node


def runner_kwargs(params):
    return dict(params)


def get_runner(name, **kwargs):
    if name not in VARIANTS:
        raise ValueError("未知变体 %r（可选 %s）" % (name, " / ".join(VARIANTS)))
    if name == "ref-heavy":
        return lambda *a, **kw: reference(*a, params=kw)
    raise RuntimeError("变体 %r 的 adapter 尚未实现" % name)
