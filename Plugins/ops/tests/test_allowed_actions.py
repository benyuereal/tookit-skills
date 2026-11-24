#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""天花板约束：allowed_actions 必须被 run 真正消费。

这是薄约束的两个执行断点之一。此前 op-constrain（实际不存在）产出的
允许集合只是提示语，loop.py 从不读取 → 越界的改法照样能跑，
「memory 天花板 → 只准减字节」这条纪律在最后一步漏了。

现在的约定：来源是 `<卡>/roofline.json` 的 `bound` 字段（op-roofline
derive.py 的产出），映射成允许的动作类；候选的 `action_class` 必须在集合内。
没有 roofline.json 时不校验（不是每张卡都做过 roofline）。
"""
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]
LOOP = REPO / "Plugins/ops/skills/op-optimize/loop.py"


def _load_loop():
    spec = importlib.util.spec_from_file_location("_loop_aa", str(LOOP))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_loop_aa"] = mod
    spec.loader.exec_module(mod)
    return mod


class TestAllowedActions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import yaml  # noqa: F401
        cls.loop = _load_loop()

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.card = Path(self._td.name)
        (self.card / "bench").mkdir()
        (self.card / "bench" / "ab.py").write_text("# stub\n", encoding="utf-8")

    def tearDown(self):
        self._td.cleanup()

    def _cands(self, action_class):
        (self.card / "candidates.yaml").write_text(
            "directions:\n"
            "  - {id: occupancy, statement: t, status: open, refutes: 0}\n"
            "candidates:\n"
            "  - {id: c1, direction: occupancy, action_class: %s, status: pending}\n"
            "refuted: []\n" % action_class, encoding="utf-8")

    def _run(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = self.loop.cmd_run(SimpleNamespace(
                card=self.card, tags=["c1"], skv=None, s=None, topk=None,
                nvalid=None, heads=None, qscale=None, rounds=3, tol=None,
                primary_qscale=None, stress_mult=None, no_acc=False, force=False))
        return rc, buf.getvalue()

    def test_memory_ceiling_rejects_compute_action(self):
        self._cands("raise-issue-rate")
        (self.card / "roofline.json").write_text('{"bound":"memory"}', encoding="utf-8")
        rc, out = self._run()
        self.assertEqual(rc, 2)
        self.assertIn("不在允许集合", out)
        self.assertIn("memory", out)

    def test_memory_ceiling_allows_reduce_bytes(self):
        self._cands("reduce-bytes")
        (self.card / "roofline.json").write_text('{"bound":"memory"}', encoding="utf-8")
        rc, out = self._run()
        self.assertNotIn("不在允许集合", out)

    def test_compute_ceiling_rejects_reduce_bytes(self):
        self._cands("reduce-bytes")
        (self.card / "roofline.json").write_text('{"bound":"compute"}', encoding="utf-8")
        rc, out = self._run()
        self.assertEqual(rc, 2)
        self.assertIn("compute", out)

    def test_no_roofline_json_means_no_enforcement(self):
        """没做过 roofline 分析的卡不该被拦。"""
        self._cands("raise-issue-rate")
        rc, out = self._run()
        self.assertNotIn("不在允许集合", out)

    def test_undeclared_action_class_blocks(self):
        (self.card / "candidates.yaml").write_text(
            "directions:\n"
            "  - {id: occupancy, statement: t, status: open, refutes: 0}\n"
            "candidates:\n"
            "  - {id: c1, direction: occupancy, status: pending}\n"
            "refuted: []\n", encoding="utf-8")
        (self.card / "roofline.json").write_text('{"bound":"memory"}', encoding="utf-8")
        rc, out = self._run()
        # 语义已定：未声明 action_class **拦**（不是警告后放行）。
        # 不拦等于让"没填"绕过天花板约束。
        self.assertEqual(rc, 2, out)
        self.assertIn("未声明 action_class", out)

    def test_latency_ceiling_maps_to_pipeline_actions(self):
        self._cands("persistent-dispatch")
        (self.card / "roofline.json").write_text('{"bound":"latency"}', encoding="utf-8")
        rc, out = self._run()
        self.assertNotIn("不在允许集合", out)


if __name__ == "__main__":
    unittest.main()
