#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scripts/forge_step.py —— 「造变体」这一环的判据。

重点全在**反向**：这条链路的失败模式不是「跑不起来」，而是
「跑起来了但算错 / 慢得莫名」，而这两种都不报错。

已踩过的三处（都写进用例）：
  ① VEC 当运行期变量 → 寄存器数组溢出到私有内存（private 0→528B，慢 3.3x）；
  ② choose_params 取「第一个能整除」→ 自动生成版比手工版慢 24%；
  ③ H 与 THREADS*8*VEC 不匹配 → 越界读，不报错，只算错。
"""
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "forge_step.py"


def _load():
    spec = importlib.util.spec_from_file_location("_forge_step_t", str(SCRIPT))
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


F = _load()


# ---------------------------------------------------------------- 选参

def test_choose_params_prefers_vec2_at_4096():
    """H=4096 必须选实测最优的 THREADS=256/VEC=2，不是 512/VEC=1。

    反向意义：512/VEC=1 也「能整除、能编译、能跑」，只是慢 24%。
    没有这条，退化完全静默。
    """
    p = F.choose_params(4096)
    assert p == {"THREADS": 256, "WARPS": 4, "VEC": 2}, p


def test_choose_params_rejects_unsupported_H():
    """**反向**：匹配不上就报错，不许兜底（兜底会产出算错的变体）。"""
    with pytest.raises(F.ForgeStepError) as e:
        F.choose_params(100)
    assert "不提供兜底" in str(e.value)


def test_param_basis_declares_its_own_limit():
    """选参依据必须自带 model_limit —— 一个形状上的结论不能当全类规律。"""
    assert "model_limit" in F.PARAM_BASIS
    assert len(F.PARAM_BASIS["measured"]) >= 2, "至少两个实测点才能比较"


# ---------------------------------------------------------------- 体检

def test_health_check_rejects_private_spill():
    """private>0 ⇒ 寄存器数组溢出。实测慢 3.3 倍，且**不报错**。"""
    ok, probs = F.health_check({"private": 528, "vgpr": 40, "threads": 256}, 256)
    assert not ok
    assert "私有内存" in probs[0]


def test_health_check_rejects_threads_mismatch():
    """threads 与 __launch_bounds__ 不符 ⇒ 编译器静默截断。"""
    ok, probs = F.health_check({"private": 0, "vgpr": 40, "threads": 128}, 256)
    assert not ok
    assert "launch_bounds" in probs[0]


def test_health_check_passes_clean_profile():
    """正向：干净的 profile 必须通过（否则上面两条是假绿）。"""
    ok, probs = F.health_check({"private": 0, "vgpr": 38, "threads": 256}, 256)
    assert ok and probs == []


# ---------------------------------------------------------------- 全链路（要 GPU 编译器）

@pytest.mark.skipif(not (Path("/opt/dtk/bin/hipcc").is_file()
                         or __import__("shutil").which("hipcc")),
                    reason="本机无 hipcc，编译判据未生效")
def test_forge_variant_end_to_end(tmp_path):
    """真生成→真编译→真体检。产物必须落在 card 下，且 private=0。"""
    card = tmp_path / "op-x"
    card.mkdir()
    r = F.forge_variant("t1", "fused-add-reduce", F.choose_params(4096), 4096,
                        None, card=card)
    assert Path(r["so"]).is_file()
    assert Path(r["adapter"]).is_file()
    assert r["profile"]["private"] == 0, r["profile"]
    assert r["profile"]["threads"] == 256
    # 目录必须一致：adapter 里写的是相对 HERE/_kernels/xxx.so
    assert Path(r["so"]).parent.name == "_kernels"
    assert Path(r["adapter"]).parent == Path(r["so"]).parent.parent


@pytest.mark.skipif(not (Path("/opt/dtk/bin/hipcc").is_file()
                         or __import__("shutil").which("hipcc")),
                    reason="本机无 hipcc，编译判据未生效")
def test_forge_failure_leaves_no_half_product(tmp_path):
    """**反向**：体检不过时必须删掉 .so，否则同名变体下次会被直接加载。"""
    card = tmp_path / "op-x"
    card.mkdir()
    # H=4096 但强行指定 VEC=1 + THREADS=512 —— 能编译，但我们要让它「体检失败」
    # 的路径可测：直接 monkeypatch health_check 返回失败。
    orig = F.health_check
    try:
        F.health_check = lambda prof, expect_threads=None: (False, ["模拟失败"])
        with pytest.raises(F.ForgeStepError):
            F.forge_variant("t2", "fused-add-reduce",
                            {"THREADS": 256, "WARPS": 4, "VEC": 2}, 4096,
                            None, card=card)
    finally:
        F.health_check = orig
    so = card / "runs" / "forged" / "_kernels" / "t2.so"
    assert not so.exists(), "体检失败后留下了 .so —— 它会被同名变体误加载"


def test_selftest_passes():
    assert F.selftest() == 0
