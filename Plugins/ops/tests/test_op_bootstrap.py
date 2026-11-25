#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scripts/op_bootstrap.py 的测试：新算子冷启动。

缺口：已有卡的链路是通的（roofline → lang_select → advance），
但**新来的算子第一环就断**——derive.py 只认 sparse-mla，没有建卡入口，
卡片 schema 有 20+ 字段且 additionalProperties=false。

本文件守的核心命题：**bootstrap 算出的 bytes 必须是分母，算错就是结论反。**
所以反向用例全部围绕「什么情况下 bytes 会算错」。
"""
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "op_bootstrap.py"


def _load():
    spec = importlib.util.spec_from_file_location("_bootstrap_t", str(SCRIPT))
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


B = _load()


def test_selftest_passes():
    assert B.selftest() == 0


# ---------------------------------------------------------------- bytes 是分母

def test_two_reads_one_write():
    """正向：2 读 1 写 = 3 × M×H×2B。**这是分母**，算错方向就反。"""
    r = B.generate("t1", "t", "elementwise", "y = a + b", "M=1024,H=4096",
                   "a:bf16[M,H], b:bf16[M,H]", "y:bf16[M,H]",
                   "elementwise-2in", write=False)
    assert abs(r["bytes_mb"] - 25.166) < 0.01
    kinds = [d["kind"] for d in r["detail"]]
    assert kinds.count("read") == 2 and kinds.count("write") == 1


def test_missing_dim_is_rejected():
    """**反向**：维度名不在 shape 里必须拒。

    否则 `x:bf16[M,H]` 在 shape={M} 下只乘 M，bytes 少一个数量级，
    分母错 → 结论方向反（本仓已复现多次）。
    """
    with pytest.raises(ValueError, match="不在形状"):
        B.generate("t2", "t", "elementwise", "y=x", "M=1024",
                   "x:bf16[M,H]", "y:bf16[M,H]", "elementwise", write=False)


def test_gemm_pattern_produces_no_bytes():
    """正向：GEMM 不吃 bytes 分母（由 FLOPs 约束）。"""
    r = B.generate("t3", "t", "gemm", "C=A@B", "M=1024,N=1024,K=1024",
                   "A:bf16[M,K], B:bf16[K,N]", "C:bf16[M,N]", "gemm", write=False)
    assert r["bytes_mb"] is None


# ---------------------------------------------------------------- 复合算子

def test_composite_operator_rejected():
    """**核心**：含矩阵乘的 math 不许用 elementwise pattern。

    实测场景：「add + RMSNorm + SwiGLU」既有 elementwise（x+r）、又有归约
    （rmsnorm）、又有两个 GEMM。硬套 elementwise 只算到 201MB，
    漏掉 FLOPs 主导的部分 —— 分母只有一半。
    """
    with pytest.raises(ValueError, match="矩阵乘"):
        B.generate("t4", "t", "activation",
                   "h = rmsnorm(x+residual); y = silu(h @ Wg) * (h @ Wu)",
                   "M=8192,H=4096", "x:bf16[M,H]", "y:bf16[M,H]",
                   "elementwise-2in", write=False)


def test_multi_mechanism_math_rejected():
    """**反向**：math 里三个以上不同机制 ⇒ 多半是复合算子，该拆。"""
    with pytest.raises(ValueError, match="复合算子"):
        B.generate("t5", "t", "activation",
                   "rmsnorm then softmax then topk then reduce",
                   "M=8,H=8", "x:bf16[M,H]", "y:bf16[M,H]",
                   "elementwise", write=False)


def test_family_pattern_mismatch_rejected():
    """**反向**：family 与访问模式不匹配 ⇒ 分母大概率错，拒。"""
    with pytest.raises(ValueError, match="不匹配"):
        B.generate("t6", "t", "gemm", "y=A@B", "M=1024,N=1024,K=1024",
                   "A:bf16[M,K], B:bf16[K,N]", "C:bf16[M,N]",
                   "elementwise", write=False)


# ---------------------------------------------------------------- 不覆盖 / 不猜

def test_refuses_to_overwrite_existing_card():
    """**反向**：bootstrap 不覆盖既有卡 —— 那会毁掉已有实测证据。"""
    with pytest.raises(ValueError, match="已存在"):
        B.generate("rmsnorm-fused", "t", "norm", "y=x", "M=8,H=8",
                   "x:bf16[M,H]", "y:bf16[M,H]", "reduce", write=True)


def test_generated_common_does_not_fake_reference():
    """**核心**：生成的 reference 必须是显式 NotImplementedError。

    参考实现是正确性的唯一锚。猜一个能让整个门禁「通过」一个算错的实现
    —— 因为两边都错。宁可让它跑不了。
    """
    src = B.build_common("t7", [{"name": "x", "dtype": "bf16", "dims": ["M"]}],
                         "y = x")
    assert "NotImplementedError" in src
    assert "正确性的唯一锚" in src


def test_parse_io_handles_commas_inside_brackets():
    """**反向**：`x:bf16[M,H], y:bf16[M,H]` 不能按逗号切分（M,H 里也有逗号）。"""
    out = B.parse_io("x:bf16[M,H], y:bf16[M,H]")
    assert len(out) == 2, out
    assert out[0]["dims"] == ["M", "H"], out[0]
    assert out[1]["name"] == "y", out[1]
