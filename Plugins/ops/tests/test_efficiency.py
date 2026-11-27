#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`scripts/efficiency.py` 的用例：算对的要算对，该拒的必须拒。

反向用例是主体。本仓的教训是「判据写进去了却从未生效」——
一个只做 `achieved / peak` 的除法器，在分母用了 1100 标称、精度配错、
placement 缺 vgpr 时都会**静默给出一个漂亮的小数**，而结论方向可能是反的
（见 constants.yaml 里「拿 1100 当分母 = 87%，别做了」vs「1680 = 66%，
还有 1.5x」）。

被测对象：<repo>/scripts/efficiency.py。
全部用例不碰 GPU、不写 perf/ 目录（临时 YAML/JSON 只落在 tmp_path）。
"""
import copy
import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "efficiency.py"
CONSTANTS = REPO / "Plugins/ops/skills/op-roofline/constants.yaml"
LADDER = REPO / "Plugins/ops/op-atlas/bw-ladder.yaml"


def _load():
    spec = importlib.util.spec_from_file_location(
        "_efficiency_under_test", str(SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


eff = _load()


# ---------------------------------------------------------------- 夹具

def _record(**over):
    """一条放置完整、可计算的 record（MBU 1108/1680）。over 覆盖顶层字段。"""
    rec = {
        "variant": "lightop-fused",
        "median_us": 242.2,
        "achieved_bw": 1108.0,
        "pattern": "rw_1to1",
        "placement": {
            "grid": [128, 1],
            "ctas_per_cu": 4,
            "vgpr": 96,
            "lds_per_cta": 16384,
            "waves_per_cu": 16,
        },
        "placement_unknown": {},
    }
    rec.update(over)
    return rec


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(textwrap.dedent(text), encoding="utf-8")
    return p


# ---------------------------------------------------------------- 正向：用真实常数验算

def test_mfu_bf16_real_constant():
    """260.5 TF / 279 = 0.934（用仓库里真实的 bf16 achievable 峰值）。"""
    r = eff.compute_mfu(260.5, "bf16")
    assert "error" not in r, r
    assert r["mfu"] == pytest.approx(260.5 / 279.0, abs=2e-6)
    assert 0.93 < r["mfu"] < 0.94
    assert r["denominator"] == {
        "id": "compute.bf16_gemm_peak_tf",
        "value": 279.0,
        "date": "2026-11-23",
        "caliber": "achievable",
        "unit": "TFLOPS",
        "source": r["denominator"]["source"],
    }
    assert r["numerator"]["achieved_flops_tflops"] == 260.5


def test_mfu_fp8_uses_579_7_not_279():
    """fp8 的分母是 579.7，不是 bf16 的 279（两个数，不可共用）。"""
    r = eff.compute_mfu(498.5, "fp8")
    assert "error" not in r, r
    assert r["mfu"] == pytest.approx(498.5 / 579.7, abs=2e-6)
    assert r["denominator"]["id"] == "compute.fp8_peak_layers.achievable_tf"
    assert r["denominator"]["value"] == 579.7
    # 若错用 279 会得到 1.79（荒谬），确认现在不是那个数
    assert r["mfu"] < 1.0


@pytest.mark.parametrize("pattern,peak", [
    ("r_only", 1870.0),
    ("w_only", 1692.0),
    ("rw_1to1", 1680.0),
    ("rw_2to1", 1730.0),
])
def test_mbu_four_ladder_tiers(pattern, peak):
    """四档各用各的分母，不取平均。"""
    r = eff.compute_mbu(peak / 2.0, pattern)
    assert "error" not in r, r
    assert r["denominator"]["value"] == peak
    assert r["mbu"] == pytest.approx(0.5, abs=2e-6)
    assert r["denominator"]["date"] == "2026-11-23"


def test_record_happy_path_reports_mbu():
    r = eff.efficiency_from_record(_record())
    assert "error" not in r, r
    assert r["mbu"] == pytest.approx(1108.0 / 1680.0, abs=2e-6)
    assert r["denominator"]["value"] == 1680.0
    # placement 一并回显，证明结论可迁移到「什么条件下」
    assert r["placement"]["vgpr"] == 96


# ---------------------------------------------------------------- 反向：分母来源

def test_denominator_without_date_rejected(tmp_path):
    """分母缺 verified_on（来源三件套里的 date）→ 拒绝。"""
    doc = eff._load_yaml(CONSTANTS, "c")
    del doc["compute"]["bf16_gemm_peak_tf"]["verified_on"]
    import yaml
    p = tmp_path / "no_date.yaml"
    p.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    r = eff.compute_mfu(260.5, "bf16", constants_path=str(p))
    assert "error" in r, r
    assert "日期" in r["error"] or "date" in r["error"]


def test_denominator_with_bad_date_format_rejected(tmp_path):
    """date 存在但格式不合法（'2026/11/23'）→ 拒绝，不当成合法来源。"""
    doc = eff._load_yaml(CONSTANTS, "c")
    doc["compute"]["bf16_gemm_peak_tf"]["verified_on"] = "2026/11/23"
    import yaml
    p = tmp_path / "bad_date.yaml"
    p.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    r = eff.compute_mfu(260.5, "bf16", constants_path=str(p))
    assert "error" in r, r
    assert "日期" in r["error"] or "date" in r["error"]


def test_denominator_nominal_caliber_rejected(tmp_path):
    """分母 caliber=nominal（规格上限）→ 拒绝当达成率分母。"""
    doc = eff._load_yaml(CONSTANTS, "c")
    doc["compute"]["bf16_gemm_peak_tf"]["caliber"] = "nominal"
    import yaml
    p = tmp_path / "nominal_caliber.yaml"
    p.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    r = eff.compute_mfu(260.5, "bf16", constants_path=str(p))
    assert "error" in r, r
    assert "achievable" in r["error"]


def test_denominator_impl_measured_caliber_rejected(tmp_path):
    """impl_measured（49.8 那条 aiter opus）不许当分母。"""
    doc = eff._load_yaml(CONSTANTS, "c")
    doc["compute"]["fp8_peak_layers"]["achievable"]["caliber"] = "impl_measured"
    import yaml
    p = tmp_path / "impl.yaml"
    p.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
    r = eff.compute_mfu(49.8, "fp8", constants_path=str(p))
    assert "error" in r, r
    assert "achievable" in r["error"]


def test_nominal_bandwidth_1100_rejected():
    """1_nominal 的 1100 是产品标称，显式拒绝（不是「未知模式」）。"""
    r = eff.compute_mbu(1108, "1_nominal")
    assert "error" in r, r
    assert "标称" in r["error"] or "nominal" in r["error"]
    # 不存在的梯子路径同样拒绝，不猜一个档
    r = eff.compute_mbu(1108, "rw_1to1", ladder_path="/no/such/ladder.yaml")
    assert "error" in r


# ---------------------------------------------------------------- 反向：精度不匹配

def test_bf16_denominator_on_fp8_rejected():
    """用 bf16 的 279 算 fp8（真实 498.5 TF）→ 1.79 > 1，必须报分母错。

    这正是「不静默给出 178%」的用例：结果里只有 error，没有 mfu 键。
    """
    r = eff.compute_mfu(498.5, "bf16")
    assert "error" in r, r
    assert "mfu" not in r
    assert "分母错" in r["error"]
    assert "1.78" in r["error"] or "1.79" in r["error"]


def test_unknown_precision_rejected():
    """未知精度不落到任何默认分母。"""
    r = eff.compute_mfu(100.0, "fp4")
    assert "error" in r, r
    assert "mfu" not in r


def test_mbu_over_one_rejected():
    """MBU > 1.0（用了错档）→ 报分母错，不当正常值。"""
    r = eff.compute_mbu(2000.0, "rw_1to1")   # 2000 > 1680
    assert "error" in r, r
    assert "分母错" in r["error"]
    assert "mbu" not in r


# ---------------------------------------------------------------- 反向：placement

def test_record_missing_vgpr_rejected():
    """placement 缺 vgpr 键 → 拒绝计算（不是补默认值）。"""
    rec = _record()
    del rec["placement"]["vgpr"]
    r = eff.efficiency_from_record(rec)
    assert "error" in r, r
    assert "vgpr" in r["error"]


def test_record_vgpr_zero_rejected():
    """vgpr=0 是错误值，不是「没填」→ 拒绝。"""
    rec = _record()
    rec["placement"]["vgpr"] = 0
    r = eff.efficiency_from_record(rec)
    assert "error" in r, r
    assert "vgpr=0" in r["error"]


@pytest.mark.parametrize("field", ["grid", "ctas_per_cu", "lds_per_cta"])
def test_record_missing_any_placement_field_rejected(field):
    """grid / ctas_per_cu / lds_per_cta 缺任何一个都拒绝。"""
    rec = _record()
    del rec["placement"][field]
    r = eff.efficiency_from_record(rec)
    assert "error" in r, r
    assert field in r["error"]


def test_record_null_placement_rejected():
    """placement 逐字段写 null（未知，带原因）→ 仍拒绝计算。"""
    rec = _record()
    rec["placement"]["vgpr"] = None
    rec["placement_unknown"] = {"vgpr": "编译器版本未知，.so 已被覆盖"}
    r = eff.efficiency_from_record(rec)
    assert "error" in r, r
    assert "vgpr" in r["error"]
    assert "拒绝" in r["error"]


def test_record_without_placement_block_rejected():
    rec = _record()
    del rec["placement"]
    r = eff.efficiency_from_record(rec)
    assert "error" in r, r
    assert "placement" in r["error"]


def test_record_missing_precision_rejected():
    """有算力但没 precision → 不知道用 279 还是 579.7，拒绝。"""
    rec = _record(pattern=None)
    rec.pop("pattern")
    rec["achieved_flops"] = 260.5
    r = eff.efficiency_from_record(rec)
    assert "error" in r, r
    assert "precision" in r["error"]


def test_record_missing_pattern_rejected():
    """有带宽但没 pattern → 不知道用哪一档，拒绝。"""
    rec = _record()
    del rec["pattern"]
    r = eff.efficiency_from_record(rec)
    assert "error" in r, r
    assert "pattern" in r["error"]


def test_record_median_and_bytes_computes_rate():
    """median_us + bytes → 现算 GB/s，再除以档位。"""
    rec = _record()
    rec.pop("achieved_bw")
    rec.pop("pattern")
    # 268.4 MB / 242.2 µs = 1108.2 GB/s
    rec.update({"bytes": 268.4e6, "pattern": "rw_1to1"})
    r = eff.efficiency_from_record(rec)
    assert "error" not in r, r
    assert r["mbu"] == pytest.approx((268.4e6 / 242.2e-6) / 1e9 / 1680.0,
                                     rel=1e-6)


# ---------------------------------------------------------------- 来源三件套完整性

def test_all_denominators_have_iso_dates():
    """每个可用的分母都必须带 id/value/date，且 date 是合法 ISO 日期。"""
    import datetime
    for prec in ("bf16", "fp8"):
        d = eff.resolve_peak_denominator(prec)
        assert d["id"] and d["value"] > 0
        assert datetime.date.fromisoformat(d["date"]).isoformat() == d["date"]
    for pat in eff.PATTERN_NAMES:
        d = eff.resolve_bw_denominator(pat)
        assert d["id"] and d["value"] > 0
        assert datetime.date.fromisoformat(d["date"]).isoformat() == d["date"]
        assert d["caliber"] == "achievable"


def test_bf16_and_fp8_denominators_differ():
    """两个精度分母是两个不同的数，278/579 不可合并。"""
    bf = eff.resolve_peak_denominator("bf16")
    fp = eff.resolve_peak_denominator("fp8")
    assert bf["value"] == 279.0
    assert fp["value"] == 579.7
    assert bf["id"] != fp["id"]


# ---------------------------------------------------------------- CLI

def test_cli_flops():
    proc = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "--flops", "260.5",
         "--precision", "bf16"],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["mfu"] == pytest.approx(260.5 / 279.0, abs=2e-6)
    assert out["denominator"]["date"] == "2026-11-23"


def test_cli_nominal_bandwidth_exits_nonzero():
    proc = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "--bw", "1108",
         "--pattern", "1_nominal"],
        capture_output=True, text=True)
    assert proc.returncode == 1
    out = json.loads(proc.stdout)
    assert "error" in out


def test_cli_record(tmp_path):
    doc = {"card": "demo", "records": [_record()]}
    p = tmp_path / "rec.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "--record", str(p), "--index", "0"],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = json.loads(proc.stdout)
    assert out["mbu"] == pytest.approx(1108.0 / 1680.0, abs=2e-6)


def test_cli_record_missing_vgpr_exits_nonzero(tmp_path):
    rec = _record()
    del rec["placement"]["vgpr"]
    p = tmp_path / "rec.json"
    p.write_text(json.dumps({"records": [rec]}), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "--record", str(p)],
        capture_output=True, text=True)
    assert proc.returncode == 1
    assert "vgpr" in json.loads(proc.stdout)["error"]


def test_selftest_passes():
    proc = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "--selftest"],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "selftest PASS" in proc.stdout
