#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`scripts/self_iteration.py` 的用例：六步「能力」分，判据必须真的会报错。

为什么单独一个文件
------------------
这个评分器和 `scorecard.py` 的 D 分量的是两件事：D 分是「卡上积累了多少
实测证据」，本评分器是「六步链路有没有接通」。两者混算会让
「建了一堆工具但一张卡没跑」和「真跑通三张卡」分清不了。

所以本文件的重点是**反向**：有文件 ≠ 有能力。
最容易误判的是第 3 步——`roofline.json` 存在但里面没有
`bottleneck_signal`，若判据只看「文件在」就会静默给分。每个反向用例都
构造缺一个字段的仓库，确认对应步骤变 ✗、分数精确扣掉该步权重。

被测对象：<repo>/scripts/self_iteration.py。
全部用例不碰 GPU、不写真实 cards/runs（临时仓库只落在 tmp_path）。
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "self_iteration.py"


def _load():
    spec = importlib.util.spec_from_file_location(
        "_self_iteration_under_test", str(SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


si = _load()


# ---------------------------------------------------------------- 夹具

def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build(root, *, import_path=True, card_signal=True, roofline="ok",
           surfaces="active", mapping="full", record="ok", stage="S7",
           record_verdict=True):
    """按开关造一个临时包根。默认 = 六步全通。"""
    root = Path(root)
    # 第 1 步需要真的 atlas_query.py —— 用仓库里的唯一真源
    src = REPO / "scripts/atlas_query.py"
    _write(root / "scripts/atlas_query.py", src.read_text(encoding="utf-8"))

    lines = ["id: foo", "op_card: Plugins/ops/skills/op-foo", "variants:",
             "- repo: lib"]
    if import_path:
        lines.append("  import_path: lib.foo")
    lines += ["last_verdict:"]
    if card_signal:
        lines += ["  bottleneck:", "    bound: memory", "    evidence:",
                  "    - kind: ablation",
                  "      reproduce: Plugins/ops/skills/op-foo/ledger.md#轮 1"]
    lines += ["progress:", "  stage: %s" % stage]
    _write(root / "Plugins/ops/op-atlas/cards/foo.yaml", "\n".join(lines) + "\n")

    if roofline != "absent":
        roof = {"bound": "memory", "remaining": 0.1088,
                "t_actual_mode": "event_isolated"}
        if roofline != "no_signal":
            roof["bottleneck_signal"] = {"bound": "memory", "kind": "ablation"}
        if roofline == "no_remaining":
            roof.pop("remaining")
        _write(root / "Plugins/ops/skills/op-foo/roofline.json",
               json.dumps(roof))

    if record != "absent":
        rec = {"card": "op-foo", "decision": "ACCEPTED_BITEXACT"}
        if record != "no_measure_mode":
            rec["measure_mode"] = "event_isolated"
        if record_verdict:
            rec["falsified"] = False
        _write(root / "Plugins/ops/skills/op-foo/runs/loop/20260101.jsonl",
               json.dumps(rec) + "\n")

    ref = root / "Plugins/ops/skills/op-optimize/references"
    if surfaces == "active":
        body = "".join("- id: s%d\n  statement: s\n  verdict: active\n" % i
                       for i in range(1, 11))
    else:  # all_dead_end：13 条但全 dead_end
        body = "".join("- id: s%d\n  statement: s\n  verdict: dead_end\n" % i
                       for i in range(1, 14))
    _write(ref / "surfaces.yaml", body)
    keys = list(si.DOMINANTS)
    if mapping == "missing":
        keys.remove("vgpr-spill")
    _write(ref / "dominant-to-surface.yaml",
           "dominant_to_surface:\n"
           + "".join("  %s:\n  - s1\n" % k for k in keys))


def _collect(tmp_path, **kw):
    root = Path(tmp_path) / "repo"
    _build(root, **kw)
    return si.collect(root), root


def _run(root, *args):
    return subprocess.run([sys.executable, "-I", str(SCRIPT), "--root", str(root)]
                          + list(args), capture_output=True, text=True)


# ---------------------------------------------------------------- 基线

def test_empty_repo_scores_zero_without_crash(tmp_path):
    """空仓库 → 0 分，且不能崩。"""
    root = Path(tmp_path) / "empty"
    root.mkdir()
    r = si.collect(root)
    assert r["score"] == 0.0
    assert r["coverage"]["all_six"] == 0
    assert all(not s["passed"] for s in r["steps"].values())


def test_weights_sum_to_one():
    """权重和必须为 1.0；否则「六步全通」不等于 10 分。"""
    assert round(sum(si.WEIGHTS.values()), 12) == 1.0
    assert set(si.WEIGHTS) == {k for k, _ in si.STEPS}


def test_full_fixture_scores_ten_and_coverage_one(tmp_path):
    """六步全通 → 10.0，且 coverage 算出 1 张全通的卡。"""
    r, _ = _collect(tmp_path)
    assert r["score"] == 10.0, {k: v["passed"] for k, v in r["steps"].items()}
    assert r["coverage"]["all_six"] == 1
    assert r["coverage"]["cards"] == ["foo"]
    assert all(s["detail"] for s in r["steps"].values()), "每步都要给明细"


# ---------------------------------------------------------------- 反向

def test_roofline_without_signal_does_not_earn_attribute(tmp_path):
    """**最容易误判的一条**：有 roofline.json 但没 bottleneck_signal，
    且卡片也没有带 reproduce 的 bottleneck → 第 3 步不得分。

    「有文件」不等于「有能力」。若判据只看文件存在，这一步会静默给分。
    """
    r, _ = _collect(tmp_path, roofline="no_signal", card_signal=False)
    assert r["steps"]["attribute"]["passed"] is False
    assert r["score"] == pytest.approx(8.0)
    assert r["coverage"]["all_six"] == 0


def test_signal_in_card_evidence_counts_even_without_roofline_signal(tmp_path):
    """OR 的另一半：roofline 没 signal，但卡片 evidence 带 reproduce → 通。"""
    r, _ = _collect(tmp_path, roofline="no_signal", card_signal=True)
    assert r["steps"]["attribute"]["passed"] is True


def test_all_dead_end_surfaces_do_not_earn_pick(tmp_path):
    """surfaces.yaml 有 13 条但全 dead_end → 第 4 步不得分。"""
    r, _ = _collect(tmp_path, surfaces="all_dead_end")
    assert r["steps"]["pick"]["passed"] is False
    assert r["score"] == pytest.approx(8.5)


def test_missing_dominant_mapping_does_not_earn_pick(tmp_path):
    """五个 dominant 缺一个 → 第 4 步不得分。"""
    r, _ = _collect(tmp_path, mapping="missing")
    assert r["steps"]["pick"]["passed"] is False
    assert "vgpr-spill" in r["steps"]["pick"]["detail"]


def test_record_without_measure_mode_does_not_earn_experiment(tmp_path):
    """record 有但没有 measure_mode → 第 5 步不得分。"""
    r, _ = _collect(tmp_path, record="no_measure_mode")
    assert r["steps"]["experiment"]["passed"] is False
    assert r["score"] == pytest.approx(8.0)


def test_measure_mode_unknown_is_not_a_declaration(tmp_path):
    """loop.py 缺失时写的是字符串 "unknown"——它不是「声明了模式」。"""
    r, root = _collect(tmp_path)
    p = root / "Plugins/ops/skills/op-foo/runs/loop/20260101.jsonl"
    rec = json.loads(p.read_text(encoding="utf-8"))
    rec["measure_mode"] = "unknown"
    p.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    r = si.collect(root)
    assert r["steps"]["experiment"]["passed"] is False


def test_stage_not_converged_does_not_earn_converge(tmp_path):
    """卡在 S4 → 第 6 步不得分。"""
    r, _ = _collect(tmp_path, stage="S4")
    assert r["steps"]["converge"]["passed"] is False
    assert r["score"] == pytest.approx(8.5)


def test_missing_import_path_does_not_earn_locate(tmp_path):
    """variants 没有 import_path → 第 1 步不得分。"""
    r, _ = _collect(tmp_path, import_path=False)
    assert r["steps"]["locate"]["passed"] is False
    assert r["score"] == pytest.approx(8.5)


def test_missing_remaining_does_not_earn_ceiling(tmp_path):
    """roofline 缺 remaining → 第 2 步不得分。"""
    r, _ = _collect(tmp_path, roofline="no_remaining")
    assert r["steps"]["ceiling"]["passed"] is False
    assert r["score"] == pytest.approx(8.5)


def test_record_without_convergence_verdict_does_not_earn_converge(tmp_path):
    """record 有 measure_mode 但没有 verdict/falsified → 第 6 步不得分。"""
    r, _ = _collect(tmp_path, record_verdict=False)
    assert r["steps"]["converge"]["passed"] is False


@pytest.mark.parametrize("kw,step,delta", [
    ({"import_path": False}, "locate", 1.5),
    ({"roofline": "no_remaining"}, "ceiling", 1.5),
    ({"roofline": "no_signal", "card_signal": False}, "attribute", 2.0),
    ({"surfaces": "all_dead_end"}, "pick", 1.5),
    ({"record": "no_measure_mode"}, "experiment", 2.0),
    ({"stage": "S4"}, "converge", 1.5),
])
def test_each_missing_field_drops_exact_weight(tmp_path, kw, step, delta):
    """反向总表：故意漏一个字段，该步变 ✗ 且只扣该步权重（10×w），
    而不是「静默给分」或「整仓归零」。"""
    r, _ = _collect(tmp_path, **kw)
    assert r["steps"][step]["passed"] is False
    assert r["score"] == pytest.approx(10.0 - delta)
    assert r["coverage"]["all_six"] == 0


# ---------------------------------------------------------------- CLI

def test_selftest_passes():
    r = subprocess.run([sys.executable, "-I", str(SCRIPT), "--selftest"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "selftest PASS" in r.stdout


def test_json_output_shape(tmp_path):
    _, root = _collect(tmp_path)
    r = _run(root, "--json")
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout)
    assert d["score"] == 10.0
    assert d["weights_sum"] == 1.0
    assert set(d["steps"]) == {k for k, _ in si.STEPS}
    assert {"all_six", "cards"} <= set(d["coverage"])


def test_human_output_has_six_lines_and_coverage(tmp_path):
    _, root = _collect(tmp_path)
    r = _run(root)
    assert r.returncode == 0, r.stderr
    assert "自我迭代能力" in r.stdout
    assert r.stdout.count("[✓]") == 6
    assert "六步全通的卡   1 张" in r.stdout


# ---------------------------------------------------------------- 真实仓库

def test_real_repo_runs_and_is_bounded():
    """真实仓库：分数在 [0,10]，六步都有明细，coverage 非负。

    不断言具体分数——分数会随仓库进展变化；但结构必须成立，
    否则「分数涨了」无法解释。
    """
    r = si.collect(REPO)
    assert 0.0 <= r["score"] <= 10.0
    assert r["weights_sum"] == 1.0
    for key, _ in si.STEPS:
        assert r["steps"][key]["detail"], "第 %s 步没有明细" % key
    assert r["coverage"]["all_six"] >= 0
    assert len(r["cards"]) >= 0


def test_real_repo_does_not_grant_experiment_without_measure_mode():
    """真实仓库的 loop record 普遍缺 measure_mode；若判据诚实，第 5 步就不能通。

    这条防的是「为了让真实仓库好看，把判据放宽成只看 record 存在」。
    它随数据变：一旦 loop.py 生成的记录都带上 measure_mode，本用例应改成
    断言通过——那才是真的进展。
    """
    r = si.collect(REPO)
    # 用评分器同一条扫描路径（_records）复核，避免判据与实现各扫各的。
    recs = si._records(REPO, si._cards(REPO))
    has_mode = any(si._record_measured(rec) for _, _, rec in recs)
    assert r["steps"]["experiment"]["passed"] == has_mode


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
