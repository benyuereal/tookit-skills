#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_means 的测试：前置推理闸门必须拦得住。

核心价值在**反向用例**：一个只放行的闸门等于没有闸门。
每条判据都造一份「该被拒」的候选，断言它真的被拒、且理由指向对应字段。
"""
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _load():
    spec = importlib.util.spec_from_file_location(
        "_check_means", str(REPO / "scripts/check_means.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _card(cands, root=None):
    import yaml
    d = Path(root or tempfile.mkdtemp())
    (d / "candidates.yaml").write_text(
        yaml.dump({"candidates": cands}, allow_unicode=True), encoding="utf-8")
    return d


FULL = {"id": "a", "status": "pending",
        "means": "把 residual 写回融进 norm kernel，少一趟 HBM",
        "predicted_effect": "-8%", "expected_bottleneck": "memory-bound"}


class TestCheckMeans(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load()

    def test_selftest_passes(self):
        self.assertEqual(self.m.selftest(), 0)

    def test_complete_manual_candidate_passes(self):
        probs, st = self.m.check_card(_card([FULL]))
        self.assertEqual(probs, [])
        self.assertEqual(st["checked"], 1)

    def test_effect_range_and_single_value_accepted(self):
        for eff in ("-6%~-10%", "0%", "-8%", "1.15x"):
            probs, _ = self.m.check_card(_card([dict(FULL, predicted_effect=eff)]))
            self.assertEqual(probs, [], eff)

    # ---- 反向用例 ----

    def test_missing_means_rejected(self):
        c = {k: v for k, v in FULL.items() if k != "means"}
        probs, _ = self.m.check_card(_card([c]))
        self.assertTrue(probs)
        self.assertEqual(probs[0]["field"], "means")
        self.assertIn("缺 means", probs[0]["why"])

    def test_placeholder_means_rejected(self):
        """占位符不是「写了」—— TODO 进队列，跑完仍然无法归因。"""
        for bad in ("TODO", "待定", "略"):
            probs, _ = self.m.check_card(_card([dict(FULL, means=bad)]))
            self.assertTrue(probs, bad)
            self.assertIn("占位符", probs[0]["why"])

    def test_short_means_rejected(self):
        probs, _ = self.m.check_card(_card([dict(FULL, means="调参")]))
        self.assertTrue(probs)
        self.assertIn("字符", probs[0]["why"])

    def test_prose_predicted_effect_rejected(self):
        """散文预期无法证伪，必须拒。"""
        probs, _ = self.m.check_card(
            _card([dict(FULL, predicted_effect="省一次写回")]))
        self.assertTrue(probs)
        self.assertEqual(probs[0]["field"], "predicted_effect")
        self.assertIn("可机检", probs[0]["why"])

    def test_invalid_bottleneck_rejected(self):
        """自造词对不上 loop.py 的天花板约束。"""
        probs, _ = self.m.check_card(
            _card([dict(FULL, expected_bottleneck="gpu-bound")]))
        self.assertTrue(probs)
        self.assertEqual(probs[0]["field"], "expected_bottleneck")
        self.assertIn("不在 surfaces.yaml", probs[0]["why"])

    def test_surfaces_class_is_accepted_as_bottleneck(self):
        """surfaces.yaml 的 class（如 memory-path）也是合法词表。"""
        probs, _ = self.m.check_card(
            _card([dict(FULL, expected_bottleneck="memory-path")]))
        self.assertEqual(probs, [])

    def test_auto_candidate_without_forge_evidence_rejected(self):
        """**核心反向**：没有证据就无法区分「真生成」和「手填冒充」。"""
        probs, _ = self.m.check_card(
            _card([{"id": "auto-1", "status": "pending",
                    "_origin": "advance-overlay"}]))
        self.assertTrue(probs)
        self.assertEqual(probs[0]["field"], "forge_evidence")

    def test_auto_candidate_with_evidence_exempt(self):
        probs, _ = self.m.check_card(
            _card([{"id": "auto-1", "status": "pending",
                    "_origin": "advance-overlay",
                    "forge_evidence": {"profile": {"threads": 256}}}]))
        self.assertEqual(probs, [], "自动生成候选免检，不许要求写 means")

    def test_non_pending_skipped(self):
        """已 adopted/verified 的不回头补 means。"""
        probs, st = self.m.check_card(
            _card([dict(FULL, status="verified", means=None)]))
        self.assertEqual(probs, [])
        self.assertEqual(st["checked"], 0)

    def test_checker_does_not_touch_files(self):
        """纯函数式：跑一遍后 candidates.yaml 字节不变。"""
        d = _card([dict(FULL, means="TODO")])
        before = (d / "candidates.yaml").read_bytes()
        self.m.check_card(d)
        self.assertEqual((d / "candidates.yaml").read_bytes(), before)

    def test_cli_exit_codes(self):
        ok = _card([FULL])
        bad = _card([dict(FULL, means=None)])
        r0 = subprocess.run([sys.executable, "-I",
                             str(REPO / "scripts/check_means.py"), str(ok)],
                            capture_output=True, text=True)
        r1 = subprocess.run([sys.executable, "-I",
                             str(REPO / "scripts/check_means.py"), str(bad)],
                            capture_output=True, text=True)
        r2 = subprocess.run([sys.executable, "-I",
                             str(REPO / "scripts/check_means.py"),
                             str(ok / "nope")], capture_output=True, text=True)
        self.assertEqual(r0.returncode, 0, r0.stdout + r0.stderr)
        self.assertEqual(r1.returncode, 1, r1.stdout + r1.stderr)
        self.assertEqual(r2.returncode, 2, r2.stdout + r2.stderr)

    def test_json_output_is_machine_readable(self):
        import json
        bad = _card([dict(FULL, predicted_effect="省一次写回")])
        r = subprocess.run([sys.executable, "-I",
                            str(REPO / "scripts/check_means.py"), str(bad),
                            "--json"], capture_output=True, text=True)
        out = json.loads(r.stdout)
        self.assertFalse(out["ok"])
        self.assertEqual(out["problems"][0]["field"], "predicted_effect")


if __name__ == "__main__":
    unittest.main()
