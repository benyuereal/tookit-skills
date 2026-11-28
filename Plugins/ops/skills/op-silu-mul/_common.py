#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""op-silu-mul 的 adapter（bootstrap 生成，**需要人工补齐 reference 的数学**）。

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

INPUT_NAMES = ("x1", "x2",)


def build_inputs(params, seed=0, device=None):
    """构造输入。

    silu-mul 是逐元素算子，**输入分布不改变语义**（不像 softmax / quant 那样
    决定命中率或量化误差），所以这里可以真的生成，而不是留 stub。
    x2 用 randn 而非全 1：全 1 会让 `silu(x1)*x2 == silu(x1)`，
    等于把乘法那一路的访存从测量里抹掉。
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))

    def _num(v):
        return int(v) if not isinstance(v, float) else v

    M = _num(params.get("M", 8192))
    H = _num(params.get("H", 4096))
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    x1 = torch.randn((M, H), generator=g, dtype=torch.float32).to(dev).to(torch.bfloat16)
    x2 = torch.randn((M, H), generator=g, dtype=torch.float32).to(dev).to(torch.bfloat16)
    return (x1, x2)


def reference(x1, x2, params=None):
    """fp32 参考：y[i,j] = silu(x1[i,j]) * x2[i,j]

    三条纪律（与 op-gated-add 同一套）：
      * fp32 计算，**最后不转回 bf16** —— 参考是真值，转回去等于自己抬高容差；
      * 不 in-place 改输入 —— 否则基准和变体共享内存，测出来的差是脏的；
      * sigmoid 用 torch 的稳定实现，不手写 1/(1+exp(-x))（大负数会溢出）。
    """
    x1f = x1.to(torch.float32)
    x2f = x2.to(torch.float32)
    return (x1f * torch.sigmoid(x1f)) * x2f


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
