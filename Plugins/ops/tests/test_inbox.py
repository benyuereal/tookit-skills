#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""inbox.py 的反向测试。

背景（/root/atlas-handoff/WORKFLOW-UPGRADE-PLAN.md §3）：Inbox 是**事后审计**，
不是审批。它唯一能骗人的地方是把「有 gate 失败的产出」标成「不需要人看」，
所以本文件重点验：
  A. 有任何一个 gate=fail → needs_human 必须被强制 yes（无参数可绕过）
  B. INBOX 只追加，旧行一个字节都不变
  C. 同 id 重复追加 → 拒绝
  D. 未知 gate 名 / 非法取值 → 拒绝（写错名字不等于通过）
"""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _load():
    spec = importlib.util.spec_from_file_location(
        "_inbox", str(REPO / "scripts/inbox.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ALL_PASS = "E41=pass,E42=pass,E43=pass,pytest=pass,floor_recompute=pass"


class TestInboxEntry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load()

    def build(self, gates, **kw):
        g = self.m.parse_gates([gates]) if isinstance(gates, str) else gates
        kw.setdefault("ts", "2026-11-24T02:10:11")
        return self.m.build_entry("rmsnorm-fused", "S6", "一句话结论", g,
                                  kw.pop("artifacts", []), **kw)

    def test_all_pass_needs_human_no(self):
        e = self.build(ALL_PASS)
        self.assertEqual(e["needs_human"], "no")
        self.assertEqual(e["id"], "2026-11-24T02:10:11/rmsnorm-fused/S6")
        self.assertEqual(e["gates"]["E43"], "pass")

    def test_any_fail_forces_yes(self):
        e = self.build("E41=pass,E42=fail")
        self.assertEqual(e["needs_human"], "yes",
                         "有 fail 时 needs_human 必须被强制 yes")

    def test_single_fail_forces_yes(self):
        e = self.build("pytest=fail")
        self.assertEqual(e["needs_human"], "yes")

    def test_invalid_gate_name_rejected(self):
        with self.assertRaises(ValueError):
            self.m.parse_gates(["E4x=pass"])

    def test_invalid_gate_value_rejected(self):
        with self.assertRaises(ValueError):
            self.m.parse_gates(["E41=ok"])

    def test_empty_gates_rejected(self):
        with self.assertRaises(ValueError):
            self.m.parse_gates([])

    def test_invalid_stage_rejected(self):
        with self.assertRaises(ValueError):
            self.m.build_entry("x", "S9", "c", {"E41": "pass"}, [], "t")

    def test_claim_newline_rejected(self):
        with self.assertRaises(ValueError):
            self.m.build_entry("x", "S6", "两行\n结论", {"E41": "pass"}, [], "t")


class TestInboxFile(unittest.TestCase):
    def setUp(self):
        self.m = _load()
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "INBOX.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def _entry(self, ts="2026-11-24T02:10:11", gates=None, card="c1", stage="S6"):
        g = self.m.parse_gates([gates or ALL_PASS])
        return self.m.build_entry(card, stage, "结论", g, [], ts)

    def test_append_is_append_only(self):
        first = self._entry(ts="2026-11-24T02:00:00")
        self.m.append_entry(self.path, first)
        before = self.path.read_bytes()
        second = self._entry(ts="2026-11-24T03:00:00")
        self.m.append_entry(self.path, second)
        after = self.path.read_bytes()
        self.assertTrue(after.startswith(before), "追加后旧字节必须原样保留")

    def test_duplicate_id_rejected(self):
        e = self._entry()
        self.m.append_entry(self.path, e)
        with self.assertRaises(self.m.DuplicateId):
            self.m.append_entry(self.path, e)
        lines = [l for l in self.path.read_text().splitlines() if l.strip()]
        self.assertEqual(len(lines), 1, "重复追加不许产生第二行")

    def test_list_needs_human_filters(self):
        self.m.append_entry(self.path, self._entry(ts="A", gates=ALL_PASS))
        self.m.append_entry(self.path,
                            self._entry(ts="B", gates="E42=fail",
                                        card="c2"))
        allrows = self.m.list_entries(self.path)
        self.assertEqual(len(allrows), 2)
        pending = self.m.list_entries(self.path, needs_human=True)
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["gates"]["E42"], "fail")

    def test_missing_file_reads_as_empty(self):
        self.assertEqual(self.m.list_entries(self.path), [],
                         "文件不存在应视作空，不是错误")

    def test_bad_json_line_is_error_not_skipped(self):
        self.path.write_text('{"id":"a","card":"c","stage":"S6"}\nnot json\n')
        with self.assertRaises(ValueError):
            self.m.read_entries(self.path)

    def test_cli_append_then_list(self):
        rc = self.m.main(["--inbox", str(self.path), "append",
                          "--card", "c1", "--stage", "S7",
                          "--claim", "ok", "--gates", ALL_PASS,
                          "--ts", "2026-11-24T04:00:00"])
        self.assertEqual(rc, 0)
        row = json.loads(self.path.read_text().strip())
        self.assertEqual(row["card"], "c1")
        self.assertEqual(row["needs_human"], "no")
        out = self.m.main(["--inbox", str(self.path), "list", "--needs-human"])
        self.assertEqual(out, 0)

    def test_cli_duplicate_exit_nonzero(self):
        args = ["--inbox", str(self.path), "append", "--card", "c1",
                "--stage", "S6", "--claim", "ok", "--gates", ALL_PASS,
                "--ts", "T"]
        self.assertEqual(self.m.main(args), 0)
        self.assertNotEqual(self.m.main(args), 0, "重复 id 的退出码必须非 0")


if __name__ == "__main__":
    unittest.main()
