#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scripts/ingest.py 的独立测试（补上 agent 漏交的那一份）。

吸纳类工具的失败模式是「抽错了还静默通过」，不是崩溃。所以判据全在：
  · provenance 的 quote 必须**真能在原文找到**（防编造）
  · 抽不到必须标 missing，**不许用 0/null 冒充**
  · human 字段一个都不许被机器填（防「假装机器判定」）
  · 同一源目录跑两次结果一致（防随机性）
"""
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "ingest.py"
SCHEMA = REPO / "Plugins/ops/ingest/schema.yaml"
MHC = Path("/workspace/mhc_kernels")


def _load():
    spec = importlib.util.spec_from_file_location("_ingest_t", str(SCRIPT))
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


I = _load()
NEED = pytest.mark.skipif(not MHC.is_dir(), reason="本机无 mhc_kernels 语料")


def test_selftest_passes():
    assert I.selftest() == 0


@pytest.fixture(scope="module")
def recs():
    return I.extract(MHC)


@NEED
def test_extracts_enough_records(recs):
    assert recs, "在真实语料上一条都没抽到"
    assert len(recs) > 20, "只抽到 %d 条，明显偏少" % len(recs)


@NEED
def test_every_provenance_quote_is_real(recs):
    """**核心**：每条 provenance 的 quote 必须真能在该文件该行找到。

    这是防止「编造出处」的唯一机器判据。没有它，schema 只是一层好看的包装。
    """
    bad = []
    for r in recs:
        p = r["provenance"]
        f = MHC / p["file"]
        if not f.is_file():
            bad.append((r["id"], "文件不存在: %s" % p["file"]))
            continue
        lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        if not (0 < p["line"] <= len(lines)):
            bad.append((r["id"], "行号越界: %s:%s" % (p["file"], p["line"])))
            continue
        if p["quote"].strip()[:30] not in lines[p["line"] - 1]:
            bad.append((r["id"], "quote 与该行不符: %r" % p["quote"][:40]))
    assert bad == [], "provenance 造假 %d 条：%s" % (len(bad), bad[:3])


@NEED
def test_missing_is_explicit_not_zero(recs):
    """**反向**：抽不到必须写 missing 并给原因，不许用 0 或 null 冒充。"""
    seen = 0
    for r in recs:
        if "missing" not in r:
            continue
        seen += 1
        m = r["missing"]
        assert isinstance(m, dict) and m, "missing 必须是非空 dict，实际 %r" % (m,)
        for k, why in m.items():
            assert str(why).strip(), "missing 的 %s 没给原因" % k
    assert seen >= 1, "本语料没有任何 missing —— 那这条判据没被验证过（假绿）"


@NEED
def test_human_fields_never_filled_by_machine(recs):
    """**核心反向**：标记为 human 的字段不许被机器填。

    业主反复强调：能机器抽的和只能人判的必须分开。人判字段被机器填了，
    就等于把「判断」伪装成「数据」。
    """
    if not SCHEMA.is_file():
        pytest.skip("没有 schema.yaml，判据未生效")
    schema = yaml.safe_load(SCHEMA.read_text(encoding="utf-8"))
    human = {f["name"] for f in schema["fields"] if f.get("extract") == "human"}
    assert human, "schema 里没有 human 字段 —— 那说明所有判断都被伪装成机器产出了"
    for r in recs:
        overlap = human & set(r.keys())
        assert not overlap, (
            "human 字段被机器填了：%s（记录 %s）" % (overlap, r.get("id")))


@NEED
def test_deterministic(recs):
    """同一语料跑两次，记录数必须一致（防随机取样）。"""
    again = I.extract(MHC)
    assert len(again) == len(recs)
    assert [r["id"] for r in again] == [r["id"] for r in recs]
