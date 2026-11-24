#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""recall.py 的过滤/排序/降级测试。

recall 是说「上次这个 (bound, signal) 改了什么」的唯一入口，它能骗人的地方有四处，
本文件逐条钉住：

  A. 空 ledger → 「无历史记录」退出 0，不崩
  B. 过滤正确（bound / (bound,signal) / card；多条件取 AND）
  C. 坏 JSON 行 → 跳过并 warning，其余条目照常返回
  D. 缺 citation 字段 → 标 incomplete_citation，不许静默通过
  E. ts 倒序；同 ts 已证伪优先显示
  F. 输出里每条都带 citation；falsified=true 有显著标记
"""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "scripts/recall.py"


def _load():
    spec = importlib.util.spec_from_file_location("_recall", str(TOOL))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rec(card, ts, verdict="significant", bound="memory", signal="low_occupancy",
        falsified=False, **kw):
    r = {"card": card, "ts": ts, "tag": card + "-tag", "verdict": verdict,
         "direction": "fusion-boundary", "expected_effect": "-11%",
         "base_ms": 0.195, "cand_ms": 0.161, "falsified": falsified,
         "trigger_pattern": {"bound": bound, "signal": signal,
                             "direction": "fusion-boundary"}}
    r.update(kw)
    return r


def write_ledger(lines):
    """lines 里 str 原样写入（用于造坏行），dict 序列化成 JSON。"""
    d = tempfile.mkdtemp()
    p = Path(d) / "ledger.jsonl"
    p.write_text("\n".join(
        x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)
        for x in lines) + "\n", encoding="utf-8")
    return p


def run(*args):
    return subprocess.run([sys.executable, "-I", str(TOOL)] + list(args),
                          capture_output=True, text=True)


class TestRecall(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load()

    def q(self, *lines, **kw):
        p = write_ledger(lines)
        recs, warns = self.m.load(p)
        return self.m.query(recs, **kw), warns

    # ---- A 空 ledger
    def test_empty_ledger_exit_zero_and_says_no_history(self):
        p = write_ledger([])
        r = run("--ledger", str(p), "--bound", "compute")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("无历史记录", r.stdout)

    def test_missing_ledger_treated_as_empty(self):
        r = run("--ledger", "/nonexistent/ledger.jsonl")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("无历史记录", r.stdout)

    # ---- B 过滤
    def test_filter_by_bound(self):
        got, _ = self.q(
            rec("a", "2026-11-24T10:00:00", bound="memory"),
            rec("b", "2026-11-24T10:00:01", bound="compute"),
            rec("c", "2026-11-24T10:00:02", bound="memory",
                signal="wait_lds_high"), bound="memory")
        self.assertEqual(sorted(x["card"] for x in got), ["a", "c"])

    def test_filter_by_bound_and_signal_combination(self):
        got, _ = self.q(
            rec("a", "2026-11-24T10:00:00", signal="low_occupancy"),
            rec("b", "2026-11-24T10:00:01", signal="wait_lds_high"),
            rec("c", "2026-11-24T10:00:02", signal="low_occupancy"),
            bound="memory", signal="low_occupancy")
        self.assertEqual(sorted(x["card"] for x in got), ["a", "c"])

    def test_filter_by_card(self):
        got, _ = self.q(
            rec("op-rmsnorm-fused", "2026-11-24T10:00:00"),
            rec("op-softmax", "2026-11-24T10:00:01"),
            card="op-rmsnorm-fused")
        self.assertEqual([x["card"] for x in got], ["op-rmsnorm-fused"])

    def test_no_match_is_empty_list_not_error(self):
        got, _ = self.q(rec("a", "2026-11-24T10:00:00"), bound="nope")
        self.assertEqual(got, [])

    # ---- C 坏 JSON（反向）
    def test_bad_json_line_skipped_with_warning(self):
        p = write_ledger([
            rec("good", "2026-11-24T10:00:00"),
            "{this is not json",
            rec("also-good", "2026-11-24T10:00:01"),
        ])
        recs, warns = self.m.load(p)
        self.assertEqual(sorted(x["card"] for x in recs), ["also-good", "good"])
        self.assertEqual(len(warns), 1)
        self.assertIn("坏 JSON", warns[0])

    def test_bad_json_does_not_crash_cli(self):
        p = write_ledger([rec("a", "2026-11-24T10:00:00"), "{{"], )
        r = run("--ledger", str(p))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("warning", r.stderr)
        self.assertIn("命中 1 条", r.stdout)

    # ---- D incomplete citation（反向）
    def test_missing_verdict_flagged_incomplete_citation(self):
        bad = rec("a", "2026-11-24T10:00:00")
        bad.pop("verdict")
        cite, missing = self.m.citation_of(bad)
        self.assertIsNone(cite)
        self.assertEqual(missing, ["verdict"])

    def test_incomplete_citation_shown_not_silent(self):
        bad = rec("a", "2026-11-24T10:00:00")
        bad.pop("ts")
        p = write_ledger([bad])
        r = run("--ledger", str(p), "--json")
        d = json.loads(r.stdout)
        self.assertEqual(d["results"][0]["citation_status"], "incomplete_citation")
        self.assertIn("ts", d["results"][0]["citation_missing"])
        self.assertIn("incomplete_citation", r.stdout)
        human = run("--ledger", str(p))
        self.assertIn("incomplete_citation", human.stdout)

    # ---- E 排序
    def test_results_sorted_by_ts_descending(self):
        got, _ = self.q(
            rec("old", "2026-11-24T09:00:00"),
            rec("new", "2026-11-24T12:00:00"),
            rec("mid", "2026-11-24T10:30:00"))
        self.assertEqual([x["card"] for x in got], ["new", "mid", "old"])

    def test_same_ts_falsified_first(self):
        got, _ = self.q(
            rec("live", "2026-11-24T12:00:00", falsified=False),
            rec("dead", "2026-11-24T12:00:00", verdict="significant_regression",
                falsified=True))
        self.assertEqual([x["card"] for x in got], ["dead", "live"])

    # ---- F 输出契约
    def test_every_result_has_citation(self):
        p = write_ledger([rec("a", "2026-11-24T10:00:00"),
                          rec("b", "2026-11-24T10:00:01")])
        r = run("--ledger", str(p), "--json")
        d = json.loads(r.stdout)
        self.assertEqual(d["count"], 2)
        for row in d["results"]:
            self.assertEqual(row["citation_status"], "ok")
            self.assertIn(row["card"], row["citation"])
            self.assertIn(row["ts"], row["citation"])
            self.assertIn("verdict=", row["citation"])

    def test_falsified_marker_prominent(self):
        p = write_ledger([rec("dead", "2026-11-24T10:00:00", falsified=True)])
        r = run("--ledger", str(p))
        self.assertIn("⚠ 已证伪", r.stdout)

    def test_does_not_read_decision_records_without_verdict_as_verdict(self):
        """loop 写 decision、索引写 verdict：两者都要被 citation 认。"""
        r = rec("a", "2026-11-24T10:00:00")
        r.pop("verdict")
        r["decision"] = "REJECTED_NO_GAIN"
        cite, missing = self.m.citation_of(r)
        self.assertEqual(missing, [])
        self.assertIn("verdict=REJECTED_NO_GAIN", cite)

    def test_selftest_passes(self):
        r = run("--selftest")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("selftest PASS", r.stdout)


if __name__ == "__main__":
    unittest.main()
