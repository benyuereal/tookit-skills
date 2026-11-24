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
    assert {k: p[k] for k in ("THREADS", "WARPS", "VEC")} == \
        {"THREADS": 256, "WARPS": 4, "VEC": 2}, p
    # WRITE_RO 是独立的语义维度（写不写 residual_out），默认 1。
    # 单独断言而不是并进上面：两者混在一起时，一个变了另一个的断言也会假失败。
    assert p["WRITE_RO"] == 1, p


def test_choose_params_rejects_unsupported_H():
    """**反向**：匹配不上就报错，不许兜底（兜底会产出算错的变体）。"""
    with pytest.raises(F.ForgeStepError) as e:
        F.choose_params(100)
    assert "不提供兜底" in str(e.value)


def test_param_basis_declares_its_own_limit():
    """选参依据必须自带 model_limit —— 一个形状上的结论不能当全类规律。"""
    assert "model_limit" in F.PARAM_BASIS
    assert len(F.PARAM_BASIS["measured"]) >= 4, "sweep 有 4 组，依据必须是同一批实测"


def test_param_basis_conclusion_matches_measurement():
    """**核心**：结论必须能被同文件里的实测数据**验算**，不能是散文。

    反向意义：我第一版把 27% 的差距归因成「512 线程占更多调度槽」，
    而 sweep 显示真正变量是 VEC。如果结论只是散文，这个错误会一直躺着。
    这里用 measured 里的数字直接算极差，结论必须与之相符。
    """
    ms = [m["ms"] for m in F.PARAM_BASIS["measured"]]
    assert len(ms) >= 4, "依据必须覆盖完整 sweep，不是挑几个点"
    # 结论说「都不影响」，那全部四组的极差就必须落在噪声内。
    # 反过来：如果哪天又出现 24% 这种差距，这条会立刻失败，逼着重写结论。
    spread = max(ms) / min(ms) - 1
    assert spread < 0.05, (
        "四组极差 %.2f%% 超出噪声 —— 「参数都不影响结果」这个结论不成立，"
        "PARAM_BASIS 与 choose_params 都要重写" % (spread * 100))
    assert "不影响" in F.PARAM_BASIS["conclusion"] or "噪声" in F.PARAM_BASIS["conclusion"]
    # 被推翻的旧结论必须留痕 —— 否则下一个人会照着再犯一次
    assert F.PARAM_BASIS.get("refuted_claims"), "缺少 refuted_claims，旧错误会被重新引入"


def test_param_basis_is_same_batch():
    """**核心**：依据必须声明是同批测量，且带 base。

    反向意义：0.2299 ms 那次的错误根源就是「拿孤立单跑跟别的数字比」。
    不同批 = 不同 base 漂移，差值里混着机器状态，方向都可能反。
    """
    assert F.PARAM_BASIS.get("same_batch") is True, (
        "依据未声明同批测量 —— 非同批的数字不可相减")
    assert "base" in F.PARAM_BASIS, "缺 base 测量就无法判断这轮机器状态是否正常"
    assert F.PARAM_BASIS["base"]["mad"] / F.PARAM_BASIS["base"]["ms"] < 0.05, (
        "base 的 MAD/median 超过 5% 熔断线，这批数据本身不可信")


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
