#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3.1：用 fake backend 单测 driver 控制流（不碰 GPU、不调 claude）。"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "_driver_test", str(ROOT / "Plugins/ops/engine/driver.py"))
D = importlib.util.module_from_spec(_spec)
sys.modules["_driver_test"] = D
_spec.loader.exec_module(D)


def test_three_rounds_produce_complete_records():
    be = D.FakeBackend()
    recs = D.iterate(be, 3, card="fake")
    assert len(recs) == 3
    for r in recs:
        assert r.workspace_commit, "缺 commit"
        assert r.claim, "缺 claim"
        assert r.cand_ms is not None and r.base_ms is not None
    assert [r.verdict for r in recs] == ["KEEP", "KEEP", "KEEP"]
    assert recs[-1].stop_reason == "rounds_done"


def test_relaxed_revert_keeps_perf_regression_but_rolls_back_correctness():
    # 第 2 轮变慢（0.22 > 0.20）但正确性 ok → 保留，不 reset
    be = D.FakeBackend(times=[0.18, 0.22, 0.15])
    recs = D.iterate(be, 2, card="fake")
    assert recs[1].verdict == "NEUTRAL"
    assert be.reset_calls == []

    # 第 2 轮正确性破坏 → REVERT 且 reset 到上一轮 commit
    be2 = D.FakeBackend(fail_every=2)
    recs2 = D.iterate(be2, 2, card="fake")
    assert recs2[1].verdict == "REVERT"
    assert be2.reset_calls == ["c2"]  # 回滚到第 1 轮结束时的 head


def test_pre_green_is_neutral_until_first_correctness_pass():
    be = D.FakeBackend(green_from=3)
    recs = D.iterate(be, 3, card="fake")
    assert recs[0].verdict == "NEUTRAL"
    assert recs[1].verdict == "NEUTRAL"
    assert recs[2].verdict == "KEEP"


def test_plateau_stops_after_three_flat_rounds():
    be = D.FakeBackend(flat=True)          # 每轮只有 0.5% 改善 < 1.5% 噪声底
    recs = D.iterate(be, 10, card="fake")
    assert len(recs) == 3, "P3.3：连续 3 轮无提升必须停"
    assert recs[-1].stop_reason == "plateau"
    assert all(r.verdict == "NEUTRAL" for r in recs)


def test_impl_fails_stops_after_consecutive_correctness_breaks():
    be = D.FakeBackend(fail_every=1)
    recs = D.iterate(be, 10, card="fake")
    assert recs[-1].stop_reason == "impl_fails"
    assert len(recs) == 3      # engine/loop.py 的 MAX_IMPL_FAILS（单一真源）


def test_prompt_contains_structured_feedback_and_refuted_directions():
    be = D.FakeBackend()
    recs = D.iterate(be, 2, card="fake")
    prompt = D.render_student_prompt(recs, "fake")
    for key in ("verdict", "vs_base", "refuted_directions", "claim"):
        assert key in prompt, "prompt 缺 %s" % key
    assert "fusion-boundary" in prompt
    # 第一轮 prompt 明确说只写实现、不要自己测
    assert "不要在会话里自己跑 bench" in D.render_student_prompt([], "fake")


def test_noop_changes_are_invalid_and_never_keep():
    """agent 没改任何 agent 路径 → INVALID_NOOP，不 KEEP、不晋升、立即停。"""
    be = D.FakeBackend(noop=True)
    recs = D.iterate(be, 3, card="fake")
    assert recs[0].verdict == "INVALID_NOOP"
    assert all(r.verdict != "KEEP" for r in recs)
    assert len(recs) == 1 and recs[-1].stop_reason == "noop"
    assert be.reset_calls == ["c1"]      # 空提交被丢弃


def test_engine_outputs_do_not_count_as_agent_changes():
    """memory/ results/ .artifacts/ 是引擎写的，不算 agent 改动。"""
    assert D.agent_changes(["memory/ledger.jsonl", "results/bench_std.json"]) == []
    assert D.agent_changes(["Plugins/ops/skills/op-x/_common.py"]) == [
        "Plugins/ops/skills/op-x/_common.py"]


def test_noise_floor_uses_same_generation_spread():
    assert D.noise_floor_from([0.99]) == D.NOISE_FLOOR          # 只有一个值 → 地板
    assert abs(D.noise_floor_from([0.99, 0.94]) - 0.05) < 1e-9  # spread 0.05 > 1.5%
    assert D.noise_floor_from([0.99, 0.985]) == D.NOISE_FLOOR   # spread 0.5% → 地板


def test_wide_spread_turns_a_gain_into_neutral():
    """同代 spread 5% 时，下一轮 2.5% 的"提升"不能算 KEEP。"""
    be = D.FakeBackend(times=[0.200, 0.190, 0.1915])
    recs = D.iterate(be, 3, card="fake")
    assert recs[1].verdict == "KEEP"        # 第一轮 ratio=1.0，第二轮 -5%
    # 此时 ratios=[1.0, 0.95]，spread=5% ⇒ 第三轮 -2.5% 落在噪声内
    assert recs[2].noise_floor >= 0.05
    assert recs[2].verdict == "NEUTRAL"
