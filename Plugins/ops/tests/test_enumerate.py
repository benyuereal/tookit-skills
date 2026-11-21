#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""loop.py enumerate 子命令的行为守卫。

enumerate 的唯一职责：把 surfaces.yaml 展开成「可挑的方向菜单」，
并标出已被方向预算封掉的那些。两条最关键的行为：
  1. 已封（refutes >= MAX_DIRECTION_REFUTES）的方向必须显示 SEALED
     且不计入「可挑」——否则 agent 会白跑一条已经证伪的路。
  2. --new 输出的片段必须能直接粘进 candidates.yaml（含 id/direction/status）。
"""
import importlib.util
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LOOP = REPO / "Plugins/ops/skills/op-optimize/loop.py"


def _load_loop():
    spec = importlib.util.spec_from_file_location("_loop_enum", str(LOOP))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_loop_enum"] = mod
    spec.loader.exec_module(mod)
    return mod


def _card(tmp, directions_yaml):
    p = Path(tmp) / "candidates.yaml"
    p.write_text(directions_yaml, encoding="utf-8")
    return Path(tmp)


class TestEnumerate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import yaml  # noqa: F401
        except ImportError:                        # pragma: no cover
            raise unittest.SkipTest("PyYAML 未安装")
        cls.loop = _load_loop()

    def _run(self, card, new=False):
        class A:
            pass
        a = A()
        a.card = card
        a.new = new
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.loop.cmd_enumerate(a)
        return buf.getvalue()

    def test_all_surfaces_listed(self):
        with tempfile.TemporaryDirectory() as tmp:
            card = _card(tmp, "directions: []\ncandidates: []\nrefuted: []\n")
            out = self._run(card)
        for s in self.loop.load_surfaces():
            self.assertIn(s["id"], out, "surface %s 未出现在输出里" % s["id"])

    def test_sealed_direction_marked_and_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            card = _card(tmp, (
                "directions:\n"
                "  - id: split-k\n"
                "    statement: t\n"
                "    status: open\n"
                "    refutes: 99\n"
                "candidates: []\n"
                "refuted: []\n"))
            out = self._run(card)
        sealed_line = [ln for ln in out.splitlines() if ln.startswith("SEALED")]
        self.assertTrue(sealed_line, "refutes 超阈值的方向必须标 SEALED")
        self.assertIn("split-k", sealed_line[0])
        self.assertIn("已封 1 条", out)

    def test_open_direction_stays_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            card = _card(tmp, "directions: []\ncandidates: []\nrefuted: []\n")
            out = self._run(card)
        self.assertIn("已封 0 条", out)
        self.assertIn("可挑 11 条", out)

    def test_new_mode_emits_pasteable_snippet(self):
        with tempfile.TemporaryDirectory() as tmp:
            card = _card(tmp, "directions: []\ncandidates: []\nrefuted: []\n")
            out = self._run(card, new=True)
        self.assertIn("- id: occupancy", out)
        self.assertIn("direction: occupancy", out)
        self.assertIn("status: pending", out)
        self.assertNotIn("SEALED", out, "--new 不该输出已封方向")


if __name__ == "__main__":
    unittest.main()
