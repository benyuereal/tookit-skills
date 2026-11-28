# -*- coding: utf-8 -*-
"""`engine/nextstep.py` 的判定顺序用例。

## 为什么值得单测

这是**唯一的策略实现**：agent（Claude Code）每次只问一句 `loop.py next`，
然后照做。策略判错一次，agent 就会往错的方向走一整轮 —— 而且它不会质疑，
因为 next 的输出就是"权威指令"。

实测踩过的两个错判（2026-11-28）：
  1. `op-fused-add-rmsnorm` 有可跑候选（reg-resident，比 lightop 快 43%），
     但因为还没有账本，被判成「先测基线」并建议去 `op_bootstrap` ——
     把能跑的竞争候选降级成"先建卡"。
  2. `op-gemm-bf16` 队列空，被判成「先测基线 + 看 queue」，
     正确答案是「出题」（explore）。

两个错的共同点：**在"没有账本"这一层设了一个兜底分支，挡住了后面更具体的路径。**
"""
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
NS = REPO / "Plugins/ops/engine/nextstep.py"


def _load():
    spec = importlib.util.spec_from_file_location("_ns_test", str(NS))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_ns_test"] = mod
    spec.loader.exec_module(mod)
    return mod


ns = _load()


# ---------------------------------------------------------------- 构造假卡

def _write(p: Path, text: str = "{}"):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def make_card(tmp_path, *, variants=("ref-heavy",), candidates=(),
              records=(), hip=()):
    """造一张最小的卡：只放 nextstep 需要读的三样东西。"""
    card = tmp_path / "op-fake"
    card.mkdir()
    src = "VARIANTS = (%s)\n" % ", ".join('"%s"' % v for v in variants)
    if hip:
        src += "HIP_VARIANTS = {%s}\n" % ", ".join('"%s": ("x.so", "f")' % v for v in hip)
    src += "\ndef get_runner(name, **kw):\n    return lambda *a, **k: None\n"
    _write(card / "_common.py", src)

    lines = []
    for c in candidates:
        lines.append("%s" % c)
    _write(card / "candidates.yaml",
           "candidates:\n" + ("".join("  - %s\n" % c for c in candidates) or "  []\n")
           + "refuted: []\n")

    if records:
        d = card / "runs" / "loop"
        d.mkdir(parents=True)
        import json
        (d / "20260101.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
            encoding="utf-8")
    return card


def rec(measure_mode="event_isolated", decision="CALIBRATION_ONLY"):
    return {"ts": "2026-01-01T00:00:00", "tag": "ref-heavy",
            "measure_mode": measure_mode, "decision": decision,
            "cand_ms": 1.0, "base_ms": 1.0}


def cand(cid, variant=None, status="pending", priority=1, **kw):
    d = {"id": cid, "status": status, "priority": priority}
    if variant:
        d["variant"] = variant
    d.update(kw)
    return d


# ---------------------------------------------------------------- 用例

def test_runnable_candidate_wins_even_without_records(tmp_path):
    """**回归**：有可跑变体时直接跑，不许先劝它去补 baseline。"""
    card = make_card(tmp_path, candidates=[cand("c1", "ref-heavy")])
    plan = ns.plan_next(card)
    assert plan["action"] == "run", plan
    assert plan["detail"]["tag"] == "c1"
    assert "run c1" in plan["cmd"]


def test_hip_variant_counts_as_runnable(tmp_path):
    """手写 HIP 变体不在 VARIANTS 里，但它是能跑的（踩过）。"""
    card = make_card(tmp_path, variants=("ref-heavy",),
                     hip=("fa-rms-mr2",),
                     candidates=[cand("c1", "fa-rms-mr2")])
    plan = ns.plan_next(card)
    assert plan["action"] == "run", plan
    assert plan["detail"]["variant"] == "fa-rms-mr2"


def test_pending_without_variant_becomes_implement(tmp_path):
    card = make_card(tmp_path, candidates=[cand("c1", None)])
    plan = ns.plan_next(card)
    assert plan["action"] == "implement", plan
    assert plan["detail"]["tag"] == "c1"
    assert "VARIANTS" in plan["cmd"]


def test_empty_queue_becomes_explore(tmp_path):
    """**回归**：队列空要说"出题"，不是"看 queue"。"""
    card = make_card(tmp_path)
    plan = ns.plan_next(card)
    assert plan["action"] == "explore", plan
    assert "explore.py" in plan["cmd"]


def test_proposed_only_becomes_implement(tmp_path):
    card = make_card(tmp_path, candidates=[cand("p1", None, status="proposed")])
    plan = ns.plan_next(card)
    assert plan["action"] == "implement", plan


def test_priority_decides_which_to_run(tmp_path):
    card = make_card(tmp_path, candidates=[
        cand("c2", "ref-heavy", priority=2),
        cand("c1", "ref-heavy", priority=1),
    ])
    plan = ns.plan_next(card)
    assert plan["detail"]["tag"] == "c1"
    assert plan["detail"]["also"] == ["c2"]


def test_back_to_back_record_is_not_claimable(tmp_path):
    """不可 claim 的口径不能当基线 —— 否则"已经有数了"是假的。"""
    card = make_card(tmp_path, candidates=[cand("c1", None)],
                     records=[rec("back_to_back")])
    plan = ns.plan_next(card)
    assert plan["n_claimable"] == 0
    assert plan["action"] == "implement", plan


def test_claimable_record_is_counted(tmp_path):
    card = make_card(tmp_path, candidates=[],
                     records=[rec("event_isolated", "ACCEPTED_BITEXACT")])
    plan = ns.plan_next(card)
    assert plan["n_claimable"] == 1


def test_render_puts_action_on_first_line(tmp_path):
    """agent 常常只读一行就行动 —— 第一行必须是下一步本身。"""
    card = make_card(tmp_path, candidates=[cand("c1", "ref-heavy")])
    txt = ns.render(ns.plan_next(card))
    first = txt.splitlines()[0]
    assert "下一步" in first
    assert "跑实验" in first
