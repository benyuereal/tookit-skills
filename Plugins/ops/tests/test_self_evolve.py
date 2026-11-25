#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scripts/self_evolve.py 的测试：自演化闭环。

背景（2026-11-25）：实测 25 条实验记录，**0 条是「以上一轮赢家为基线」**。
`advance.py` 的 replan() 只决定下一轮试哪个候选，不知道上一轮赢了什么，
也就不会把赢家变成新起点。

对照 ForgeKernel：它 gen0 只比 PyTorch 快 0.3%，靠「每代晋升」滚到累计 +29%。
我们 gen0 就做到 −45%，然后停下。

本文件守的核心命题：**不该晋升的记录一条都不许晋升，该晋升的一条都不许漏。**
两边都会出事：
  · 放宽 → 拿噪声/错实现当基线，后面整条链都是错的
  · 收紧 → 真实的 45% 收益被漏掉（实测踩过：只看 ratio_vs_base 字段名）
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "self_evolve.py"


def _load():
    spec = importlib.util.spec_from_file_location("_self_evolve_t", str(SCRIPT))
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


S = _load()


def R(**kw):
    """一条典型的 ACCEPTED 记录。"""
    base = {"decision": "ACCEPTED_NUMERICS", "significant": True,
            "measure_mode": "event_isolated", "ratio_vs_base": 0.90,
            "acc_ok": True, "base_ms": 1.0, "base_mad": 0.01,
            "cand_ms": 0.90, "tag": "v1", "ts": "2026-01-01T00:00:00"}
    base.update(kw)
    return base


def test_selftest_passes():
    assert S.selftest() == 0


# ---------------------------------------------------------------- 该拒的必须拒

@pytest.mark.parametrize("kw,reason", [
    ({"measure_mode": "back_to_back"}, "bad_mode"),
    ({"significant": False}, "not_significant"),
    ({"ratio_vs_base": 0.99}, "ratio_too_high"),
    ({"acc_ok": False}, "acc_failed"),
    ({"decision": "REJECTED_NO_GAIN"}, "not_accepted"),
])
def test_bad_records_are_rejected(kw, reason):
    """**核心反向**：五类不该晋升的记录，每一类都要被拦下并给出原因。"""
    r, why = S.best_accepted([R(**kw)])
    assert r is None, r
    assert why[reason] >= 1, why


# ---------------------------------------------------------------- 该收的必须收

def test_picks_best_ratio():
    r, _ = S.best_accepted([R(ratio_vs_base=0.95), R(ratio_vs_base=0.80)])
    assert r is not None and abs(S._ratio_of(r) - 0.80) < 1e-9


def test_ratio_falls_back_when_field_missing():
    """**核心**：没有 `ratio_vs_base` 字段也要能算出比例。

    实测踩过：`op-fused-add-rmsnorm/bench/ab.py` 不写这个字段
    （七份分叉的后果），只看字段名会把 −45% 的收益漏掉。
    """
    r, _ = S.best_accepted([R(ratio_vs_base=None, cand_ms=0.55, base_ms=1.0)])
    assert r is not None, "没有 ratio_vs_base 就判无收益 —— 会漏掉真实收益"
    assert abs(S._ratio_of(r) - 0.55) < 1e-9


def test_ratio_falls_back_to_delta_pct():
    """第三级兜底：只有 delta_pct 时也能算。"""
    r = {"delta_pct": -45.1}
    assert abs(S._ratio_of(r) - 0.549) < 1e-6


# ---------------------------------------------------------------- 噪声地板

def test_noise_floor_minimum_with_few_samples():
    f, why = S.noise_floor([R()])
    assert f == S.NOISE_FLOOR_MIN
    assert "不足 3" in why


def test_noise_floor_uses_measured_spread():
    """**必须是本代实测的 spread，不是常数。**

    D4 实测：同一份代码在四张卡上 MAD 从 4.9% 漂到 7.2%。
    用固定值会在噪声大的机器上过严、在安静的机器上过松。
    """
    recs = [R(base_mad=0.005), R(base_mad=0.05), R(base_mad=0.09)]
    f, _ = S.noise_floor(recs)
    assert abs(f - 0.085) < 1e-6, f


def test_noise_floor_never_below_min():
    recs = [R(base_mad=0.001), R(base_mad=0.002), R(base_mad=0.003)]
    f, _ = S.noise_floor(recs)
    assert f == S.NOISE_FLOOR_MIN, f


# ---------------------------------------------------------------- 停止判据

def test_stops_when_gain_below_floor():
    stop, why = S.should_stop([(2, 0.99)], floor=0.02)
    assert stop and "噪声地板" in why


def test_continues_when_gain_above_floor():
    stop, _ = S.should_stop([(2, 0.90)], floor=0.02)
    assert not stop


def test_stops_when_no_confirmed_best():
    """**反向**：这一代没有确认最优必须停 —— 否则会无限跑。"""
    stop, why = S.should_stop([(3, None)], floor=0.02)
    assert stop and "没有确认的最优" in why


# ---------------------------------------------------------------- 端到端

def test_seal_creates_manifest_with_evidence(tmp_path):
    """封存物必须带齐证据 —— 少一样，下一代比的就不是同一个东西。"""
    card = tmp_path / "op-t"
    (card / "runs" / "loop").mkdir(parents=True)
    (card / "kernels").mkdir()
    (card / "kernels" / "v1.hip").write_text("// fake", encoding="utf-8")
    recs = [R(tag="v1", cand_ms=0.55, base_ms=1.0)]
    (card / "runs" / "loop" / "x.jsonl").write_text(
        "\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")

    dest, mf = S.seal(card, emit=lambda *_: None)
    assert dest is not None and dest.is_dir()
    assert mf["gen"] == 1
    ev = mf["evidence"]
    for k in ("measured_ms", "ratio", "measure_mode"):
        assert ev.get(k) is not None, k
    assert mf["files"]["source"]["sha256"], "封存物缺 sha256"
    assert dest.name == "gen1"


def test_seal_refuses_to_overwrite(tmp_path):
    """**反向**：封存物一旦定下就是历史，不许覆盖。"""
    card = tmp_path / "op-t"
    (card / "runs" / "loop").mkdir(parents=True)
    (card / "runs" / "loop" / "x.jsonl").write_text(
        json.dumps(R()) + "\n", encoding="utf-8")
    assert S.seal(card, emit=lambda *_: None)[0] is not None
    dest, why = S.seal(card, emit=lambda *_: None)
    # 第二次会封成 gen2（因为 gen1 已存在），不是覆盖
    assert dest is not None and dest.name == "gen2", why


def test_seal_is_atomic_on_failure(tmp_path):
    """**反向**：封存失败不许留半成品目录。

    实测踩过：第一版直接 mkdir，后面抛异常留下半截 gen1，
    下次调用变 gen2 —— 编号错乱且看不出原因。
    """
    card = tmp_path / "op-t"
    (card / "runs" / "loop").mkdir(parents=True)
    rec = R()
    rec["cand_ms"] = "not-a-number"      # 故意制造失败
    (card / "runs" / "loop" / "x.jsonl").write_text(
        json.dumps(rec) + "\n", encoding="utf-8")
    try:
        S.seal(card, emit=lambda *_: None)
    except Exception:
        pass
    sealed = card / "reference" / "sealed"
    if sealed.is_dir():
        left = [d.name for d in sealed.iterdir()]
        assert not any(n.startswith(".tmp-") for n in left), \
            "失败后留下了临时目录：%s" % left


def test_verify_catches_tampered_file(tmp_path):
    """**反向**：封存物被改过，verify 必须发现。"""
    card = tmp_path / "op-t"
    (card / "runs" / "loop").mkdir(parents=True)
    (card / "kernels").mkdir()
    k = card / "kernels" / "v1.hip"
    k.write_text("// original", encoding="utf-8")
    (card / "runs" / "loop" / "x.jsonl").write_text(
        json.dumps(R()) + "\n", encoding="utf-8")
    dest, _ = S.seal(card, emit=lambda *_: None)
    assert dest is not None
    # 篡改封存后的副本
    (dest / "v1.hip").write_text("// tampered", encoding="utf-8")
    probs = S.verify(card, emit=lambda *_: None)
    assert any("sha256" in p for p in probs), probs


def test_plan_creates_target(tmp_path):
    card = tmp_path / "op-t"
    (card / "runs" / "loop").mkdir(parents=True)
    (card / "kernels").mkdir()
    (card / "kernels" / "v1.hip").write_text("// fake", encoding="utf-8")
    (card / "runs" / "loop" / "x.jsonl").write_text(
        json.dumps(R(tag="v1", cand_ms=0.55, base_ms=1.0)) + "\n", encoding="utf-8")
    S.seal(card, emit=lambda *_: None)
    p, out = S.plan_next(card, step=1.05, emit=lambda *_: None)
    assert p is not None and p.is_file(), out
    assert out["target_ms"] < out["base_ms"]
    assert out["next_gen"] == 2
