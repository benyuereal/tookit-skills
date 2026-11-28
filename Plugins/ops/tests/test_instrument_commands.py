#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""命令模板（playbook）的接受标准：占位符、转义、缺参报错。"""
import importlib.util
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _load():
    spec = importlib.util.spec_from_file_location(
        "_instr_cmd", str(REPO / "Plugins/ops/engine/instrument/__init__.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestCommandTemplate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load()

    def test_status_quotes_paths_with_spaces(self):
        got = self.m.command("status", loop="/a b/loop.py", card="/c")
        self.assertEqual(got, "python3 -I '/a b/loop.py' -C /c status")

    def test_run_appends_raw_extra(self):
        got = self.m.command("run", loop="/l.py", card="/c", tag="t",
                             extra=" --bench-std /x/y.json")
        self.assertEqual(got, "python3 -I /l.py -C /c run t --bench-std /x/y.json")

    def test_explore(self):
        got = self.m.command("explore", repo="/r", card="/c")
        self.assertEqual(got, "python3 -I /r/scripts/explore.py --card /c --apply")

    def test_missing_key_is_loud(self):
        with self.assertRaises(KeyError):
            self.m.command("run", loop="/l.py", card="/c", tag="t")

    def test_unknown_template_is_loud(self):
        with self.assertRaises(KeyError):
            self.m.command("no-such-command")
