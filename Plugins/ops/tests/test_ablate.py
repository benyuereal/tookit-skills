#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`scripts/ablate.py` 的判据守卫：该报错时必须报错。

重点全是**反向** —— 消融最容易退化成「凑一个份额数字」：
  · 全不显著却给出 share（硬凑）
  · measure_mode 不可 claim 却进合计（跨口径）
  · 冠军本身不显著（没有收益可归因）
  · spec 引用了没测过的 tag（把不存在当成 0）
  · share 合计越界却不报 unexplained
每个用例只落 tmp_path，不跑 GPU、不写仓库。
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "ablate.py"


def _load():
    spec = importlib.util.spec_from_file_location("_ablate_under_test", str(SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


abl = _load()


def rec(tag, ms, mad=0.001, mode="event_isolated", base=1.0, base_mad=0.001,
        shape="prefill-8k"):
    return {"tag": tag, "base_ms": base, "base_mad": base_mad, "cand_ms": ms, "mad": mad,
            "measure_mode": mode, "shape": {"shape": shape}}


def make_card(tmp_path, records, *, baseline="base", champion="champ", ablations=None,
              extra=None):
    """造一张最小卡：只写 runs/loop/*.jsonl 与 spec.yaml。"""
    d = tmp_path / "card" / "runs" / "loop"
    d.mkdir(parents=True, exist_ok=True)
    (d / "20260101.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    spec = {"baseline": baseline, "champion": champion,
            "ablations": ablations or [{"id": "a", "tag": "a"}]}
    if extra:
        spec.update(extra)
    p = tmp_path / "spec.yaml"
    import yaml
    p.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return tmp_path / "card", p


def run(tmp_path, records, **kw):
    card, spec = make_card(tmp_path, records, **kw)
    return abl.build_report(card, spec)


# ---------------------------------------------------------------- 正向

def test_two_items_half_half_is_ok(tmp_path):
    rep = run(tmp_path, [rec("base", 1.0), rec("champ", .9), rec("a", .95), rec("b", .95)],
              ablations=[{"id": "a", "tag": "a"}, {"id": "b", "tag": "b"}])
    assert rep["status"] == "ok"
    assert abs(rep["sum_share"] - 1.0) < 1e-9
    assert abs(rep["items"][0]["attribution_share"] - 0.5) < 1e-9


def test_share_denominator_is_champion_gain(tmp_path):
    """分母必须是冠军相对基准的总收益，而非「显著 delta 之和」——
    用后者时 share 恒为 1，[0.8,1.2] 校验永远通过（判据自废）。"""
    rep = run(tmp_path, [rec("base", 1.0), rec("champ", .9), rec("a", .95)])
    assert rep["gain_ms"] == pytest.approx(0.1)
    assert rep["items"][0]["delta_ms"] == pytest.approx(0.05)
    assert rep["items"][0]["attribution_share"] == pytest.approx(0.5)


# ---------------------------------------------------------------- 反向

def test_all_insignificant_never_invents_a_number(tmp_path):
    """全不显著 → no_significant_attribution，且 sum_share 必须是 None（不硬凑）。"""
    rep = run(tmp_path, [rec("base", 1.0), rec("champ", .9),
                         rec("a", .9005), rec("b", .8995)],
              ablations=[{"id": "a", "tag": "a"}, {"id": "b", "tag": "b"}])
    assert rep["status"] == "no_significant_attribution"
    assert rep["sum_share"] is None
    assert all(it["attribution_share"] is None for it in rep["items"])


def test_share_below_08_is_unexplained(tmp_path):
    rep = run(tmp_path, [rec("base", 1.0), rec("champ", .9), rec("a", .92)])
    assert rep["status"] == "unexplained"
    assert rep["sum_share"] < 0.8


def test_share_above_12_is_unexplained(tmp_path):
    """各项重叠（加总超过总收益）也是问题：归因必须满足可加性。"""
    rep = run(tmp_path, [rec("base", 1.0), rec("champ", .9), rec("a", .98), rec("c", .98)],
              ablations=[{"id": "a", "tag": "a"}, {"id": "c", "tag": "c"}])
    assert rep["status"] == "unexplained"
    assert rep["sum_share"] > 1.2


def test_back_to_back_is_not_claimable_and_excluded(tmp_path):
    rep = run(tmp_path, [rec("base", 1.0), rec("champ", .9),
                         rec("a", .98, mode="back_to_back")])
    it = rep["items"][0]
    assert it["claimable"] is False
    assert it["attribution_share"] is None
    assert "measure_mode" in it["not_claimable_reason"]
    assert rep["sum_share"] is None


def test_champion_not_significant_has_nothing_to_attribute(tmp_path):
    rep = run(tmp_path, [rec("base", 1.0), rec("champ", .999), rec("a", .95)])
    assert rep["status"] == "no_significant_attribution"
    assert rep["champion_problems"]
    assert rep["items"][0]["claimable"] is False


def test_champion_slower_than_baseline_is_rejected(tmp_path):
    rep = run(tmp_path, [rec("base", 1.0), rec("champ", 1.05), rec("a", .95)])
    assert rep["status"] == "no_significant_attribution"
    assert any("慢于基准" in p for p in rep["champion_problems"])


def test_missing_tag_raises_spec_error(tmp_path):
    """spec 引用了没测过的 tag → SpecError（退出码 2），不许当成 0。"""
    with pytest.raises(abl.SpecError) as ei:
        run(tmp_path, [rec("base", 1.0), rec("champ", .9), rec("a", .95)],
            ablations=[{"id": "a", "tag": "a"}, {"id": "x", "tag": "nope"}])
    assert "不存在" in str(ei.value)


def test_shape_mismatch_is_not_claimable(tmp_path):
    """跨形状相减没有意义（分母都不同）。"""
    rep = run(tmp_path, [rec("base", 1.0), rec("champ", .9),
                         rec("a", .98, shape="decode-m1")])
    it = rep["items"][0]
    assert it["claimable"] is False
    assert "shape" in it["not_claimable_reason"]


def test_baseline_drift_makes_item_not_claimable(tmp_path):
    """两次测量对着不同的 baseline（漂移）= 漂移会被算成「这项贡献」。"""
    rep = run(tmp_path, [rec("base", 1.0), rec("champ", .9),
                         rec("a", .98, base=1.05)])
    it = rep["items"][0]
    assert it["claimable"] is False
    assert "baseline" in it["not_claimable_reason"]


def test_missing_mad_is_not_treated_as_zero(tmp_path):
    """缺 MAD ≠ MAD=0：当 0 会让任意 delta 都显著（把没证据读成证据）。"""
    r = rec("a", .98)
    r.pop("mad")
    rep = run(tmp_path, [rec("base", 1.0), rec("champ", .9), r])
    assert rep["items"][0]["claimable"] is False


def test_noise_dir_not_used(tmp_path):
    """runs/loop/noise/ 是隔离区（未形成决策），不能当归因证据。"""
    card, spec = make_card(tmp_path, [rec("base", 1.0), rec("champ", .9)])
    nd = card / "runs" / "loop" / "noise"
    nd.mkdir()
    (nd / "n.jsonl").write_text(json.dumps(rec("a", .95)) + "\n", encoding="utf-8")
    with pytest.raises(abl.SpecError):
        abl.build_report(card, spec)


# ---------------------------------------------------------------- 边界 / CLI

def test_three_times_mad_is_the_hard_line(tmp_path):
    """2.9×MAD 不显著、3.5×MAD 显著 —— 判据来自 loop._significant，不是本脚本另算。"""
    lo = run(tmp_path / "lo", [rec("base", 1.0), rec("champ", .9), rec("a", .9029)])
    hi = run(tmp_path / "hi", [rec("base", 1.0), rec("champ", .9), rec("a", .9035)])
    assert lo["items"][0]["significant"] is False
    assert hi["items"][0]["significant"] is True


def test_reuses_loop_significant_by_path():
    """判据必须来自 loop._significant（单一真源），本脚本不得内联 3×MAD。"""
    assert callable(abl._LOOP._significant)
    src = SCRIPT.read_text(encoding="utf-8")
    assert "k * max" not in src and "3.0 * max" not in src


def test_selftest_passes():
    assert abl.selftest() == 0


def test_cli_exit_codes(tmp_path):
    card, spec = make_card(tmp_path, [rec("base", 1.0), rec("champ", .9), rec("a", .92)])
    assert abl.main(["--card", str(card), "--spec", str(spec)]) == 1        # unexplained
    with pytest.raises(SystemExit) as ei:                                  # 用法错误
        abl.main(["--card", str(card)])
    assert ei.value.code == 2


def test_cli_missing_tag_exits_2_not_1(tmp_path):
    """spec 写错必须退 2：退 1 会被读成「判据跑完了、结论无效」，实验等于白做。"""
    card, spec = make_card(tmp_path, [rec("base", 1.0), rec("champ", .9)],
                           ablations=[{"id": "a", "tag": "nope"}])
    assert abl.main(["--card", str(card), "--spec", str(spec)]) == 2
