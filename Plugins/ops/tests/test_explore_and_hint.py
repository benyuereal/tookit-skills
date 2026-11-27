# -*- coding: utf-8 -*-
"""`loop.py` status 的「下一步」提示 + `explore.py` 的出题纪律。

背景（2026-11-27）：C 臂在三臂测试里唯一一次执行 loop.py 是
`loop.py -C op-moe-topk-gating status`，读到「待做 0 / 共记录 0 条」就走了，
再没碰过引擎。这条链断在「引擎没告诉对方下一步能做什么」。

这批用例锁两件事：
  1. status 在没有可跑候选时必须给出**可执行的**下一步，不能只报"没有"；
  2. explore 生成的候选必须是 `proposed`，不是 `pending` —— 变体还没实现就登记成
     待做，等于让一堵没有门的墙假装有门。
"""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
LOOP = REPO / "Plugins/ops/engine/loop.py"
EXPLORE = REPO / "scripts/explore.py"


def _load_loop():
    spec = importlib.util.spec_from_file_location("_loop_under_test_ns", str(LOOP))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_loop_under_test_ns"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def loop():
    return _load_loop()


def cands(**kw):
    d = {"candidates": [], "refuted": [], "stalled": [], "directions": [], "defaults": {}}
    d.update(kw)
    return d


# ------------------------------------------------------------ next_step_hint
def test_hint_silent_when_something_is_runnable(loop):
    c = cands(candidates=[{"id": "a", "status": "pending", "direction": "occupancy"}])
    assert loop.next_step_hint(c, [{"id": "a"}], "card")[0] == ""


def test_hint_explains_budget_block_when_pending_but_blocked(loop):
    c = cands(candidates=[{"id": "a", "status": "pending", "direction": "occupancy"}])
    h, needs = loop.next_step_hint(c, [], "card")
    assert "方向预算" in h and needs is False


def test_hint_lists_proposed_candidates(loop):
    c = cands(candidates=[
        {"id": "auto-occupancy-1", "status": "proposed", "priority": 1,
         "hypothesis": "cta_per_cu<1"},
        {"id": "auto-grid-tail-2", "status": "proposed", "priority": 2,
         "hypothesis": "rounds>1"},
    ])
    h, needs = loop.next_step_hint(c, [], "my-card")
    assert "拟做" in h and "auto-occupancy-1" in h
    assert "实现变体" in h
    assert needs is False


def test_hint_points_at_explore_when_totally_empty(loop):
    h, needs = loop.next_step_hint(cands(), [], "Plugins/ops/skills/op-x")
    assert needs is True
    assert "explore.py" in h
    assert "Plugins/ops/skills/op-x" in h
    # 必须说清"拟做不是待做"，否则下一轮的 agent 会以为可以直接 run
    assert "拟做" in h and "不是待做" in h


# --------------------------------------------------------------- explore
@pytest.fixture(scope="module")
def flash_mla():
    p = REPO / "Plugins/ops/skills/op-flash-mla"
    if not (p / "shapes.yaml").is_file():
        pytest.skip("没有 op-flash-mla 卡")
    return p


def _explore_json(card, shape=None):
    args = [sys.executable, "-I", str(EXPLORE), "--card", str(card), "--json"]
    if shape:
        args += ["--shape", shape]
    p = subprocess.run(args, capture_output=True, text=True, timeout=300,
                       cwd=str(REPO))
    assert p.returncode == 0, p.stderr[-2000:]
    return json.loads(p.stdout)


def test_explore_derives_roofline_without_roofline_json(flash_mla):
    """不要求先有 roofline.json —— 地板本来就该从模型推，不是人写的。"""
    assert not (flash_mla / "roofline.json").is_file()
    r = _explore_json(flash_mla, "decode-m32")
    roof = r["signals"].get("roof")
    assert roof, r["signals"].get("missing")
    assert roof["cost_bound"] in ("memory", "compute")
    assert roof["floors_us"]["memory"] > 0


def test_explore_proposals_are_proposed_not_pending(flash_mla):
    r = _explore_json(flash_mla, "prefill-8k")
    assert r["proposals"], r["skipped"]
    for p in r["proposals"]:
        assert p["status"] == "proposed", "拟做/待做必须分开，否则没有变体的候选会被当成能跑的"
        assert not p.get("variant"), "变体都还没有，不该写 variant"


def test_explore_direction_comes_from_surfaces(flash_mla):
    """方向不许发明：必须能在 surfaces.yaml 里找到。"""
    import yaml
    surfs = set()
    p = REPO / "Plugins/ops/skills/op-optimize/references/surfaces.yaml"
    if p.is_file():
        d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        ss = d.get("surfaces") if isinstance(d, dict) else d
        surfs = {s.get("id") for s in ss or [] if isinstance(s, dict)}
    r = _explore_json(flash_mla, "prefill-8k")
    for prop in r["proposals"]:
        if surfs:
            assert prop["direction"] in surfs, prop["direction"]


def test_explore_never_invents_a_gain(flash_mla):
    """生成的候选不许编百分比：expect 一律 0%，理由另写。"""
    r = _explore_json(flash_mla, "prefill-8k")
    for p in r["proposals"]:
        assert p["expect"] == "0%", p
        assert p["expect_basis"], "必须写清是哪个信号推的"


def test_explore_first_proposal_is_calibration_when_unmeasured(flash_mla):
    """没有实测时必须先出「量基线」这条，否则空白卡永远自锁。"""
    r = _explore_json(flash_mla, "decode-m32")
    assert r["signals"].get("achieved_us") is None
    first = r["proposals"][0]
    assert first.get("purpose") == "calibration", r["proposals"]


def test_explore_geometry_drives_different_shape_choices(flash_mla):
    """不同档位的几何不同 ⇒ 出的题也该不同，不能一套模板套所有档位。"""
    a = _explore_json(flash_mla, "prefill-8k")
    b = _explore_json(flash_mla, "decode-m32")
    da = {p["direction"] for p in a["proposals"]}
    db = {p["direction"] for p in b["proposals"]}
    assert da != db, (da, db)
