#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""消融流水线（`engine/ablate.py`）的编排守卫。

只测**编排**，不测归因判据（那在 `scripts/ablate.py` / `test_ablate.py`）：
  · preflight 三条判据各失败一次（记录缺失 / 口径不可 claim / 方法数 < 2）
  · ablation_tags / build_spec 的 tag 与 share
  · run_ablation 对 N 个 method 各 eval 一次，绝不启动 agent、不调 ablate.py
每个用例只落 tmp_path，不跑 GPU、不写仓库。
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location(
    "_ablate_pipeline_test", str(ROOT / "Plugins/ops/engine/ablate.py"))
A = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = A
_SPEC.loader.exec_module(A)


class FakeBackend:
    """最小 fake：只记录 eval / agent 调用；不跑 GPU、不调 subprocess。"""

    def __init__(self, results=None, acc_ok=True):
        self.eval_calls: list[str] = []
        self.agent_calls: list[str] = []
        self.results = list(results or [])
        self.acc_ok = acc_ok

    def eval(self, tag: str) -> dict:
        self.eval_calls.append(tag)
        i = len(self.eval_calls) - 1
        ms = self.results[i] if i < len(self.results) else 1.0
        return {"cand_ms": ms, "base_ms": 2.0, "mad": 0.001,
                "acc_ok": self.acc_ok, "measure_mode": "event_isolated"}

    def run_agent(self, prompt: str, role: str = "student") -> str:
        self.agent_calls.append(prompt)
        raise AssertionError("消融编排不许启动 agent")


def _record(tag, cand_ms=1.0, mode="event_isolated", acc_ok=True, **extra):
    r = {"tag": tag, "cand_ms": cand_ms, "base_ms": 2.0, "mad": 0.001,
         "base_mad": 0.001, "measure_mode": mode, "acc_ok": acc_ok,
         "shape": {"shape": "s"}, "ts": "2026-11-28T00:00:00"}
    r.update(extra)
    return r


def make_card(tmp_path, methods, records=None, card_name="card"):
    card = Path(tmp_path) / card_name
    (card / "runs" / "loop").mkdir(parents=True, exist_ok=True)
    (card / "candidates.yaml").write_text(
        yaml.safe_dump({"methods": methods}, allow_unicode=True), encoding="utf-8")
    if records:
        (card / "runs" / "loop" / "20261128.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
            encoding="utf-8")
    return card


METHODS2 = [{"id": "block-rows", "desc": "多行驻留"},
            {"id": "tail", "desc": "尾块处理"}]


# --------------------------------------------------------------------- preflight
def test_preflight_missing_champion_record(tmp_path):
    card = make_card(tmp_path, METHODS2)
    probs = A.preflight_ablate(card, "champ")
    assert len(probs) == 1
    assert "没有 run 记录" in probs[0]


def test_preflight_rejects_non_event_isolated(tmp_path):
    card = make_card(tmp_path, METHODS2,
                     [_record("champ", mode="back_to_back")])
    probs = A.preflight_ablate(card, "champ")
    assert len(probs) == 1
    assert "event_isolated" in probs[0]
    assert "不可 claim" in probs[0]


def test_preflight_rejects_single_method(tmp_path):
    card = make_card(tmp_path, [{"id": "only-one"}], [_record("champ")])
    probs = A.preflight_ablate(card, "champ")
    assert len(probs) == 1
    assert "方法数 < 2（当前 1）" in probs[0]


def test_preflight_passes_when_all_three_hold(tmp_path):
    card = make_card(tmp_path, METHODS2, [_record("champ")])
    assert A.preflight_ablate(card, "champ") == []


def test_preflight_reads_artifacts_tree(tmp_path):
    """记录在主路径 `.artifacts/**/runs/loop/*.jsonl` 时也要能读到。"""
    card = Path(tmp_path) / "card"
    card.mkdir()
    (card / "candidates.yaml").write_text(
        yaml.safe_dump({"methods": METHODS2}), encoding="utf-8")
    d = Path(tmp_path) / ".artifacts" / "local" / "card_run" / "runs" / "loop"
    d.mkdir(parents=True)
    (d / "20261128.jsonl").write_text(
        json.dumps(_record("champ", card="card")) + "\n", encoding="utf-8")
    assert A.preflight_ablate(card, "champ") == []


# --------------------------------------------------------------------- tags / spec
def test_ablation_tags_from_strings_and_dicts(tmp_path):
    methods = ["block-rows", {"id": "tail", "desc": "尾块"}]
    assert A.ablation_tags("champ", methods) == ["champ-no-block-rows", "champ-no-tail"]
    # 无 id 的条目丢弃，不写半截 tag
    assert A.ablation_tags("champ", ["a", {"desc": "no id"}]) == ["champ-no-a"]


def test_build_spec_share_from_records(tmp_path):
    records = {
        "champ": _record("champ", cand_ms=1.0),
        "champ-no-block-rows": _record("champ-no-block-rows", cand_ms=1.2),
        "champ-no-tail": _record("champ-no-tail", cand_ms=1.1, acc_ok=False),
    }
    spec = A.build_spec("champ", METHODS2, records)
    assert spec["champion"] == "champ"
    assert spec["baseline"] == "base"          # 无 base 记录 → 约定名
    assert [a["tag"] for a in spec["ablations"]] == [
        "champ-no-block-rows", "champ-no-tail"]
    assert all(a["derived_from"] == "champion" for a in spec["ablations"])
    a, b = spec["ablations"]
    assert a["numerically_valid"] is True
    assert b["numerically_valid"] is False     # acc_ok=False 如实透传
    assert a["share"] == pytest.approx(1.2)    # cand_ms / champion_ms
    assert b["share"] == pytest.approx(1.1)
    assert a["delta_ms"] == pytest.approx(0.2)


def test_build_spec_accepts_list_and_detects_baseline(tmp_path):
    records = [_record("base", cand_ms=2.0), _record("champ", cand_ms=1.0),
               _record("champ-no-block-rows", cand_ms=1.5)]
    spec = A.build_spec("champ", METHODS2, records)
    assert spec["baseline"] == "base"
    # 未测到的 tag → fail-closed：不可当数值有效，也没有 share
    tail = spec["ablations"][1]
    assert tail["numerically_valid"] is False
    assert tail["share"] is None


def test_build_spec_explicit_baseline_override():
    spec = A.build_spec("champ", METHODS2, {}, baseline_tag="vendor")
    assert spec["baseline"] == "vendor"
    assert all(a["share"] is None for a in spec["ablations"])


# --------------------------------------------------------------------- 编排
def test_run_ablation_evals_each_method_once_without_agent(tmp_path, monkeypatch):
    card = make_card(tmp_path, METHODS2)
    be = FakeBackend(results=[1.2, 1.1])
    # 任何子进程（尤其 ablate.py）都不许被调起
    monkeypatch.setattr("subprocess.run",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("run_ablation 不许调子进程")))
    out = A.run_ablation(be, card, "champ", METHODS2)
    assert be.eval_calls == ["champ-no-block-rows", "champ-no-tail"]
    assert be.agent_calls == []
    assert out["tags"] == be.eval_calls
    assert set(out["records"]) == set(out["tags"])
    assert all(r["tag"] == t for t, r in out["records"].items())
    assert len(out["spec"]["ablations"]) == 2


def test_run_ablation_uses_champion_record_for_share(tmp_path):
    card = make_card(tmp_path, METHODS2, [_record("champ", cand_ms=1.0)])
    be = FakeBackend(results=[1.2, 1.1])
    out = A.run_ablation(be, card, "champ", METHODS2)
    shares = [a["share"] for a in out["spec"]["ablations"]]
    assert shares == [pytest.approx(1.2), pytest.approx(1.1)]
    # 返回的 records 只含消融 tag（冠军记录只用于 spec）
    assert "champ" not in out["records"]


def test_run_ablation_empty_methods_is_a_noop(tmp_path):
    card = make_card(tmp_path, [])
    be = FakeBackend()
    out = A.run_ablation(be, card, "champ", [])
    assert be.eval_calls == [] and be.agent_calls == []
    assert out == {"tags": [], "spec": {"baseline": "base", "champion": "champ",
                                        "ablations": []}, "records": {}}


# --------------------------------------------------------------------- prompt
def test_ablate_prompt_renders_methods_and_tags(tmp_path):
    card = make_card(tmp_path, METHODS2)
    p = A.ablate_prompt(card, "champ", METHODS2)
    assert "冠军 = `champ`" in p
    assert "这 2 项方法" in p
    assert "`block-rows`：多行驻留" in p
    assert "`champ-no-block-rows`" in p and "`champ-no-tail`" in p
    assert "只删这一项" in p
    assert "VARIANTS" in p and "不要测性能" in p
