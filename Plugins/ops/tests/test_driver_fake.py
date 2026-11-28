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
    assert len(recs) == D.MAX_NOOP and recs[-1].stop_reason == "noop"
    assert be.reset_calls == ["c1", "c1"]      # 空提交被丢弃


def test_engine_outputs_do_not_count_as_agent_changes():
    """memory/ results/ .artifacts/ 是引擎写的，不算 agent 改动。"""
    assert D.agent_changes(["memory/ledger.jsonl", "results/bench_std.json"]) == []
    assert D.agent_changes(["Plugins/ops/skills/op-x/_common.py"]) == [
        "Plugins/ops/skills/op-x/_common.py"]


def test_noise_floor_uses_same_generation_spread():
    assert D.noise_floor_from([0.99]) == D.NOISE_FLOOR          # 只有一个值 → 地板
    assert abs(D.noise_floor_from([0.99, 0.94]) - 0.05) < 1e-9  # spread 0.05 > 1.5%
    assert D.noise_floor_from([0.99, 0.985]) == D.NOISE_FLOOR   # spread 0.5% → 地板


def test_repeat_spread_raises_noise_floor():
    """R1：噪声底来自**同 job 重复测量**的 spread，不是跨轮比值。"""
    # 同一次 job 重复 3 次，spread=5% ⇒ 2.5% 的改善判 NEUTRAL
    be = D.FakeBackend(times=[0.192, 0.192, 0.192], noise=0.05)
    recs = D.iterate(be, 2, card="fake")
    assert recs[0].noise_floor == 0.05
    assert recs[0].verdict == "NEUTRAL"
    # 对照：噪声 0 → 5% 改善判 KEEP
    be2 = D.FakeBackend(times=[0.190], noise=0.0)
    assert D.iterate(be2, 1, card="fake")[0].verdict == "KEEP"


def test_docs_and_config_are_not_agent_changes():
    """接受标准：只改 ledger.md → INVALID_NOOP；改 _common.py → 非空。"""
    assert D.agent_changes(["Plugins/ops/skills/op-x/ledger.md"]) == []
    assert D.agent_changes(["Plugins/ops/skills/op-x/variants.yaml"]) == []
    assert D.agent_changes(["Plugins/ops/skills/op-x/candidates.yaml"]) == []
    assert D.agent_changes(["Plugins/ops/skills/op-x/_common.py"]) == [
        "Plugins/ops/skills/op-x/_common.py"]
    assert D.agent_changes(["Plugins/ops/skills/op-x/kernels/a.hip",
                            "Plugins/ops/skills/op-x/bench/ab.py"]) != []


def test_agent_touching_engine_is_tampering():
    """P1-4：agent 改 engine/ 下的文件 = 改"用来测它的尺子" → TAMPERED + 回滚 + 停。"""
    be = D.FakeBackend(tamper=True)
    recs = D.iterate(be, 3, card="fake")
    assert len(recs) == 1
    assert recs[0].verdict == "TAMPERED"
    assert recs[0].stop_reason == "tampered"
    assert be.reset_calls == ["c1"]


def test_no_gain_is_not_a_correctness_failure():
    """relaxed-revert：没收益 ≠ 实现坏了；只有 REJECTED_ACCURACY 才回滚。"""
    assert D.correctness_of_decision("REJECTED_NO_GAIN") is True
    assert D.correctness_of_decision("INCONCLUSIVE") is True
    assert D.correctness_of_decision("CALIBRATION_ONLY") is True
    assert D.correctness_of_decision("ACCEPTED_BITEXACT") is True
    assert D.correctness_of_decision("REJECTED_ACCURACY") is False
    assert D.correctness_of_decision("ERROR") is False
    assert D.correctness_of_decision(None) is False


def test_queue_empty_is_no_work_without_spawning_agent():
    """P0-2：没有未实现候选 → NO_WORK + queue_empty，且**不调 agent**。"""
    be = D.FakeBackend(work=False)
    recs = D.iterate(be, 3, card="fake")
    assert len(recs) == 1
    assert recs[0].verdict == "NO_WORK"
    assert recs[0].stop_reason == "queue_empty"
    assert be.agent_calls == [], "队列空时不许起 agent 子进程"


def test_realbackend_exposes_system_prompt_contract(tmp_path):
    """全局角色契约必须通过 env 传给 agent wrapper。"""
    be = D.RealBackend(tmp_path, "Plugins/ops/skills/x", "tag", "true", gpu=0)
    assert be.env["OPS_WS"] == str(tmp_path.resolve())
    assert be.env["OPS_SYSTEM_PROMPT_FILE"].endswith("system_student.md")
    assert be.env["OPS_ENV_FILE"]
