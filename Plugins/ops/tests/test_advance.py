#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`scripts/advance.py` 的编排逻辑用例。

重点是**反向**：advance 只许编排，不许自己判定、不许自己跑 bench。
最容易悄悄退化的三处：
  ① 图省事直接调 bench/ab.py，绕过 loop.py 的六道门；
  ② judge 自己算显著性，和 loop.py 的 decision 打架；
  ③ precondition 全挂还继续 run（把「不该跑」当成「没通过就顺延」）。

每个反向用例都构造对应形态的临时卡（只落 tmp_path），并断言
advance 实际发出的命令集合。全部用例不碰 GPU。
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "advance.py"


def _load():
    spec = importlib.util.spec_from_file_location("_advance_under_test", str(SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


adv = _load()


# ---------------------------------------------------------------- 夹具

def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_card(tmp_path, *,
              candidates=None,
              roofline=None,
              surfaces=None):
    """造一张最小卡。candidates/roofline/surfaces 都可按用例覆写。"""
    card = tmp_path / "op-fixture"
    card.mkdir(parents=True, exist_ok=True)
    if candidates is None:
        candidates = {
            "directions": [{"id": "d1", "status": "open", "refutes": 0}],
            "candidates": [{"id": "c1", "direction": "d1", "status": "pending",
                            "priority": 1, "ev": 0.8}],
            "refuted": [],
        }
    _write(card / "candidates.yaml", json.dumps(candidates))
    if roofline is not None:
        _write(card / "roofline.json", json.dumps(roofline))
    if surfaces is not None:
        _write(tmp_path / "surfaces.yaml", json.dumps(surfaces))
    return card


@pytest.fixture
def surfaces_file(tmp_path, monkeypatch):
    """把模块级 SURFACES_PATH 指到临时文件，避免依赖仓库真实表。"""
    p = tmp_path / "surfaces.yaml"
    _write(p, "[]")
    monkeypatch.setattr(adv, "SURFACES_PATH", p)
    return p


def set_surfaces(tmp_path, rows):
    _write(tmp_path / "surfaces.yaml", json.dumps(rows))


class FakeRunner:
    """替代真实执行器：记录命令，按配置返回 rc，可按需模拟 loop.py 写记录。"""

    def __init__(self, precondition_rc=0, loop_rc=0, on_run=None):
        self.precondition_rc = precondition_rc
        self.loop_rc = loop_rc
        self.on_run = on_run
        self.commands = []
        self.runs = []
        self.precondition_calls = 0

    def precondition(self, cmd, cwd):
        adv._guard_no_bench(cmd)          # 与真实 Runner 同一道护栏
        self.precondition_calls += 1
        self.commands.append(cmd)
        rc = self.precondition_rc(cmd) if callable(self.precondition_rc) \
            else self.precondition_rc
        return rc, ""

    def loop_run(self, card, tag):
        cmd = adv.loop_run_cmd(card, tag)
        adv._guard_no_bench(cmd)
        self.commands.append(" ".join(cmd))
        self.runs.append(tag)
        rc = self.loop_rc(tag) if callable(self.loop_rc) else self.loop_rc
        if rc == 0 and self.on_run:
            self.on_run(card, tag)
        return rc, ""


def _append_record(card, tag, decision):
    p = Path(card) / "runs" / "loop" / "20260101.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"card": Path(card).name, "tag": tag,
                             "decision": decision,
                             "ts": "2026-01-01T00:00:00"}) + "\n")


def _run(card, rounds=1, runner=None, dry_run=False):
    return adv.run_advance(card, rounds, runner=runner, dry_run=dry_run,
                           emit=lambda *a, **k: None)


# ---------------------------------------------------------------- selftest

def test_selftest_passes():
    assert adv.selftest() == 0


# ---------------------------------------------------------------- 反向：门

def test_all_preconditions_fail_blocks_without_run(tmp_path, surfaces_file):
    """precondition_check 全失败 → blocked，一次 run 都不发生。"""
    set_surfaces(tmp_path, [
        {"id": "d1", "expected_gain": "50%", "precondition_check": "check-d1"},
        {"id": "d2", "expected_gain": "40%", "precondition_check": "check-d2"},
    ])
    card = make_card(tmp_path, candidates={
        "directions": [{"id": "d1", "status": "open"}, {"id": "d2", "status": "open"}],
        "candidates": [
            {"id": "c1", "direction": "d1", "status": "pending", "ev": 0.8},
            {"id": "c2", "direction": "d2", "status": "pending", "ev": 0.7},
        ],
        "refuted": [],
    })
    r = FakeRunner(precondition_rc=1)
    rep = _run(card, rounds=3, runner=r)
    assert rep["status"] == "blocked"
    assert rep["rounds"] == 1
    assert r.runs == []                      # 关键：没有进 run
    assert rep["progress_stage"] == "blocked"


def test_max_rounds_zero_exits_immediately(tmp_path, surfaces_file):
    """--max-rounds 0 → 立即退出，不跑任何东西（含 precondition）。"""
    card = make_card(tmp_path)
    r = FakeRunner()
    rep = _run(card, rounds=0, runner=r)
    assert rep["status"] == "noop"
    assert rep["rounds"] == 0
    assert r.commands == []
    assert r.precondition_calls == 0


def test_at_floor_stops_before_planning(tmp_path, surfaces_file):
    """remaining < 0.10 → at_floor，计划都还没开始。"""
    card = make_card(tmp_path, roofline={"bound": "memory", "remaining": 0.05})
    r = FakeRunner()
    rep = _run(card, rounds=2, runner=r)
    assert rep["status"] == "stopped"
    assert rep["reason"] == "at_floor"
    assert rep["rounds"] == 0


# ---------------------------------------------------------------- 反向：不绕六道门

def test_dry_run_never_calls_loop(tmp_path, surfaces_file):
    """--dry-run 跳过 run：不能出现任何 loop.py 调用。"""
    card = make_card(tmp_path)
    r = FakeRunner()
    rep = _run(card, rounds=2, runner=r, dry_run=True)
    assert r.runs == []
    assert not any("engine/loop.py" in c for c in r.commands), r.commands
    assert rep["status"] == "done"


def test_commands_never_contain_ab_py(tmp_path, surfaces_file):
    """**关键反向**：advance 发出的任何命令都不含 ab.py / bench。"""
    card = make_card(tmp_path)
    r = FakeRunner(on_run=lambda c, t: _append_record(c, t, "ACCEPTED_BITEXACT"))
    _run(card, rounds=1, runner=r)
    assert r.commands, "应当至少发过 loop.py 命令"
    joined = " ".join(r.commands).lower()
    assert "ab.py" not in joined
    assert "bench/" not in joined
    # 且唯一的实验入口是 engine/loop.py
    assert any("Plugins/ops/engine/loop.py" in c for c in r.commands)


def test_guard_rejects_bench_command():
    with pytest.raises(SystemExit):
        adv._guard_no_bench("python3 bench/ab.py --json x")


def test_loop_run_cmd_targets_engine_loop():
    cmd = adv.loop_run_cmd(Path("/tmp/card"), "c1")
    s = " ".join(cmd)
    assert "Plugins/ops/engine/loop.py" in s
    assert s.endswith("run c1")
    assert "ab.py" not in s


def test_loop_gate_rejection_falls_through_to_next_candidate(tmp_path, surfaces_file):
    """loop.py 对 top1 返回 rc=2（被自己的门拒）→ 顺延，不自己重判。"""
    card = make_card(tmp_path, candidates={
        "directions": [{"id": "d1", "status": "open"}],
        "candidates": [
            {"id": "c1", "direction": "d1", "status": "pending", "ev": 0.9},
            {"id": "c2", "direction": "d1", "status": "pending", "ev": 0.8},
        ],
        "refuted": [],
    })
    r = FakeRunner(loop_rc=lambda tag: 2 if tag == "c1" else 0,
                   on_run=lambda c, t: _append_record(c, t, "ACCEPTED_BITEXACT"))
    rep = _run(card, rounds=1, runner=r)
    assert r.runs[:2] == ["c1", "c2"]
    assert rep["decisions"] == ["ACCEPTED_BITEXACT"]


def test_all_loop_rejections_block(tmp_path, surfaces_file):
    """全部候选被 loop.py 拒 → blocked，原因写明六道门生效。"""
    card = make_card(tmp_path)
    r = FakeRunner(loop_rc=2)
    rep = _run(card, rounds=1, runner=r)
    assert rep["status"] == "blocked"
    assert "六道门" in rep["reason"]


# ---------------------------------------------------------------- judge / record

def test_judge_reads_decision_from_loop_output(tmp_path, surfaces_file):
    """judge 只读 loop.py 写的 decision，不自己算显著性。"""
    card = make_card(tmp_path)
    r = FakeRunner(on_run=lambda c, t: _append_record(c, t, "REJECTED_NO_GAIN"))
    rep = _run(card, rounds=1, runner=r)
    assert rep["decisions"] == ["REJECTED_NO_GAIN"]


def test_successful_round_advances_stage_and_records(tmp_path, surfaces_file):
    """唯一验收：跑完一轮，记录多一条，stage 前进。"""
    card = make_card(tmp_path)
    before = len(adv.read_records(card))
    r = FakeRunner(on_run=lambda c, t: _append_record(c, t, "ACCEPTED_BITEXACT"))
    rep = _run(card, rounds=1, runner=r)
    after = len(adv.read_records(card))
    assert after == before + 1
    assert rep["status"] == "done"
    assert rep["progress_stage"] == "replanned"
    prog = json.loads(adv.progress_path(card).read_text())
    assert prog["stage"] == "replanned"
    assert prog["round"] == 1


def test_dry_run_writes_no_state(tmp_path, surfaces_file):
    card = make_card(tmp_path)
    _run(card, rounds=1, runner=FakeRunner(), dry_run=True)
    assert not adv.progress_path(card).exists()


def test_dry_run_ignores_stale_terminal_state(tmp_path, surfaces_file):
    """上一次真跑留下 blocked，dry-run 不能被它卡住、也不许读它当进度。"""
    card = make_card(tmp_path)
    _write(adv.progress_path(card),
           json.dumps({"card": "op-fixture", "round": 0, "stage": "blocked",
                       "history": []}))
    rep = _run(card, rounds=1, runner=FakeRunner(), dry_run=True)
    assert rep["progress_stage"] == "replanned"
    # 且不覆写磁盘状态
    assert json.loads(adv.progress_path(card).read_text())["stage"] == "blocked"


def test_terminal_state_resets_on_new_real_invocation(tmp_path, surfaces_file):
    """blocked 是单次调用终态：重新真跑应从 idle 重开并推进到 replanned。"""
    card = make_card(tmp_path)
    _write(adv.progress_path(card),
           json.dumps({"card": "op-fixture", "round": 0, "stage": "blocked",
                       "history": []}))
    r = FakeRunner(on_run=lambda c, t: _append_record(c, t, "ACCEPTED_BITEXACT"))
    rep = _run(card, rounds=1, runner=r)
    assert rep["progress_stage"] == "replanned"


def test_consecutive_refuted_stops(tmp_path, surfaces_file):
    """连续 2 轮 refuted → 停。"""
    card = make_card(tmp_path)
    r = FakeRunner(on_run=lambda c, t: _append_record(c, t, "REJECTED_NO_GAIN"))
    rep = _run(card, rounds=5, runner=r)
    assert rep["status"] == "stopped"
    assert rep["reason"] == "连续 refuted"
    assert rep["rounds"] == 2


# ---------------------------------------------------------------- plan

def test_plan_sorts_by_expected_gain(tmp_path, surfaces_file):
    set_surfaces(tmp_path, [
        {"id": "d1", "expected_gain": "≤15%"},
        {"id": "d2", "expected_gain": "20~60%"},
    ])
    card = make_card(tmp_path, candidates={
        "directions": [{"id": "d1"}, {"id": "d2"}],
        "candidates": [
            {"id": "low", "direction": "d1", "status": "pending"},
            {"id": "high", "direction": "d2", "status": "pending"},
        ],
        "refuted": [],
    })
    top = adv.build_plan(card)
    assert [c.id for c in top] == ["high", "low"]
    assert abs(top[0].ev - 0.40) < 1e-9
    assert abs(top[1].ev - 0.15) < 1e-9


def test_plan_ignores_non_pending(tmp_path, surfaces_file):
    card = make_card(tmp_path, candidates={
        "directions": [{"id": "d1"}],
        "candidates": [
            {"id": "done", "direction": "d1", "status": "adopted"},
            {"id": "todo", "direction": "d1", "status": "pending"},
        ],
        "refuted": [],
    })
    assert [c.id for c in adv.build_plan(card)] == ["todo"]


# ---------------------------------------------------------------- replan 各分支

def test_replan_no_new_direction_stops():
    r = adv.replan(adv.Top([adv.Cand("a", "a", 0.8)]), adv.Top([]))
    assert r["action"] == "STOP" and "无可挖" in r["reason"]


def test_replan_ev_collapse_stops():
    r = adv.replan(adv.Top([adv.Cand("a", "a", 0.8)]),
                   adv.Top([adv.Cand("b", "b", 0.29)]))
    assert r["action"] == "STOP" and "0.3" in r["reason"]


def test_replan_ev_halved_stops():
    r = adv.replan(adv.Top([adv.Cand("a", "a", 0.8)]),
                   adv.Top([adv.Cand("b", "b", 0.39)]))
    assert r["action"] == "STOP" and "一半" in r["reason"]


def test_replan_same_top_continues():
    r = adv.replan(adv.Top([adv.Cand("a", "a", 0.5)]),
                   adv.Top([adv.Cand("a", "a", 0.5)]))
    assert r["action"] == "CONTINUE_SAME"


def test_replan_not_leading_1_5x_is_stalled():
    """新 top1 没领先 top2 1.5× → STALLED（不是 CONTINUE）。"""
    r = adv.replan(adv.Top([adv.Cand("a", "a", 0.5)]),
                   adv.Top([adv.Cand("b", "b", 0.50), adv.Cand("c", "c", 0.40)]))
    assert r["action"] == "STALLED"
    assert "抖动" in r["reason"]


def test_replan_switch_with_citation():
    prev = adv.Top([adv.Cand("a", "a", 0.5)])
    ledger = [{"card": "x", "ts": "2026-01-01T00:00:00", "tag": "a",
               "decision": "REJECTED_NO_GAIN"}]
    new = adv.Top([adv.Cand("b", "b", 0.9), adv.Cand("c", "c", 0.2)], ledger=ledger)
    r = adv.replan(prev, new, ledger=ledger)
    assert r["action"] == "CONTINUE_SWITCH"
    assert r["cited"] and r["cited"]["tag"] == "a"


def test_replan_citation_from_rejected_cites():
    """citation 也可来自新 top 之外的候选显式 cites 上一轮 verdict。"""
    prev = adv.Top([adv.Cand("a", "a", 0.5)])
    rejected = adv.RejectedList([adv.Cand("z", "z", 0.1,
                                          raw={"id": "z", "cites": ["a"]})])
    new = adv.Top([adv.Cand("b", "b", 0.9), adv.Cand("c", "c", 0.2)],
                  rejected=rejected)
    r = adv.replan(prev, new, ledger=[])
    assert r["action"] == "CONTINUE_SWITCH"
    assert r["cited"]["verdict"] == "cited-by-candidate"


# ---------------------------------------------------------------- 状态推进

def test_stage_is_monotonic_and_terminal_sticks():
    st = {"stage": "idle"}
    assert adv._advance_stage(st, "gated") and st["stage"] == "gated"
    assert not adv._advance_stage(st, "planned")   # 不许回退
    assert st["stage"] == "gated"
    assert adv._advance_stage(st, "blocked")
    assert not adv._advance_stage(st, "recorded")  # 终态不再前进
    assert st["stage"] == "blocked"
