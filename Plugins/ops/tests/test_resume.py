#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""resume 子命令：上下文丢了以后，下一个该跑谁。

这是「状态机 ≥ 上下文」的最小实现——所有判断从落盘状态算出来，
不依赖任何模型记忆。上下文清空、换人接手、隔天重开，跑这一条就够。

四条最关键的行为：
  1. 方向已封的候选 → blocked，不进 next（避免白跑已证伪的路）
  2. 被 claim 占着的候选 → blocked，不进 next（避免两个 worker 撞车）
  3. 跑过但不达主张门槛的 → held，不进 next（要补证据，不是重跑）
  4. 其余按 priority 排序进 next
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
LOOP = REPO / "Plugins/ops/engine/loop.py"

GOOD_CALIBER = {"what": "op_call", "alloc": True, "state": "hot",
                "shape": "s128k32768", "device": "gfx938"}


def _load_loop():
    spec = importlib.util.spec_from_file_location("_loop_resume", str(LOOP))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_loop_resume"] = mod
    spec.loader.exec_module(mod)
    return mod


class TestResume(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import yaml  # noqa: F401
        except ImportError:                    # pragma: no cover
            raise unittest.SkipTest("PyYAML 未安装")
        cls.loop = _load_loop()

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.card = Path(self._td.name)
        (self.card / "runs" / "loop").mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self._td.cleanup()

    def _write(self, text):
        (self.card / "candidates.yaml").write_text(text, encoding="utf-8")

    def _run(self, as_json=False):
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.loop.cmd_resume(SimpleNamespace(card=self.card, json=as_json))
        return buf.getvalue()

    def _data(self):
        return json.loads(self._run(as_json=True))

    # ---------------------------------------------------------------- 用例
    def test_open_candidate_goes_to_next(self):
        self._write(
            "directions:\n"
            "  - {id: occupancy, statement: t, status: open, refutes: 0}\n"
            "candidates:\n"
            "  - {id: c1, direction: occupancy, status: pending, priority: 1}\n"
            "refuted: []\n")
        d = self._data()
        self.assertEqual([x["tag"] for x in d["next"]], ["c1"])

    def test_exhausted_direction_is_blocked_not_next(self):
        self._write(
            "directions:\n"
            "  - {id: split-k, statement: t, status: open, refutes: 99}\n"
            "candidates:\n"
            "  - {id: c1, direction: split-k, status: pending}\n"
            "refuted: []\n")
        d = self._data()
        self.assertEqual(d["next"], [])
        dir_blocked = [b for b in d["blocked"] if b["tag"] == "c1"]
        self.assertEqual(len(dir_blocked), 1)
        self.assertIn("exhausted", dir_blocked[0]["reason"])

    def test_claimed_candidate_is_blocked(self):
        self._write(
            "directions:\n"
            "  - {id: occupancy, statement: t, status: open, refutes: 0}\n"
            "candidates:\n"
            "  - {id: c1, direction: occupancy, status: pending}\n"
            "refuted: []\n")
        (self.card / "runs/loop/claims").mkdir(parents=True, exist_ok=True)
        (self.card / "runs/loop/claims/c1.claim").write_text(
            json.dumps({"worker": "w2", "pid": 123}), encoding="utf-8")
        d = self._data()
        self.assertEqual(d["next"], [])
        claim_blocked = [b for b in d["blocked"] if b["tag"] == "c1"]
        self.assertEqual(len(claim_blocked), 1)
        self.assertIn("w2", claim_blocked[0]["reason"])

    def test_held_candidate_is_reported_separately(self):
        self._write(
            "directions:\n"
            "  - {id: occupancy, statement: t, status: open, refutes: 0}\n"
            "candidates:\n"
            "  - {id: c1, direction: occupancy, status: pending}\n"
            "refuted: []\n")
        # 一条缺精度的记录 → held，不该回到 next
        self.loop.append_record(self.card, {
            "ts": "2026-11-22T00:00:00", "tag": "c1", "skv": 32768,
            "base_ms": 1.0, "base_mad": 0.001, "cand_ms": 0.8, "mad": 0.001,
            "delta_pct": -20.0, "decision": "ACCEPTED_BITEXACT",
            "caliber": dict(GOOD_CALIBER), "significant": True, "acc_ok": False,
        })
        d = self._data()
        self.assertEqual(d["next"], [])
        self.assertEqual(len(d["held"]), 1)
        self.assertIn("accuracy_missing", d["held"][0]["reasons"])

    def test_priority_ordering(self):
        self._write(
            "directions:\n"
            "  - {id: occupancy, statement: t, status: open, refutes: 0}\n"
            "candidates:\n"
            "  - {id: low, direction: occupancy, status: pending, priority: 9}\n"
            "  - {id: high, direction: occupancy, status: pending, priority: 1}\n"
            "refuted: []\n")
        d = self._data()
        self.assertEqual([x["tag"] for x in d["next"]], ["high", "low"])

    def test_candidate_without_new_record_is_runnable(self):
        """从未跑过的候选当然可跑——held 只针对「跑过但不够格」。"""
        self._write(
            "directions:\n"
            "  - {id: occupancy, statement: t, status: open, refutes: 0}\n"
            "candidates:\n"
            "  - {id: never_run, direction: occupancy, status: pending}\n"
            "refuted: []\n")
        d = self._data()
        self.assertEqual([x["tag"] for x in d["next"]], ["never_run"])
        self.assertEqual(d["held"], [])

    def test_empty_card_suggests_enumerate(self):
        self._write("directions: []\ncandidates: []\nrefuted: []\n")
        out = self._run(as_json=False)
        self.assertIn("enumerate --new", out)


    def test_missing_bench_adapter_is_flagged(self):
        """冷启动最伤人的体验是「照着提示跑，立刻被拒」。

        resume 建议的 run 命令依赖卡内 A/B 适配器；缺了就明确标出来，
        不要给一条注定失败的命令。
        """
        self._write(
            "directions:\n"
            "  - {id: occupancy, statement: t, status: open, refutes: 0}\n"
            "candidates:\n"
            "  - {id: c1, direction: occupancy, status: pending}\n"
            "refuted: []\n")
        d = self._data()
        self.assertFalse(d["bench_ready"])
        self.assertTrue(any("ab.py" in b["reason"] for b in d["blocked"]))

    def test_present_bench_adapter_clears_the_flag(self):
        self._write(
            "directions:\n"
            "  - {id: occupancy, statement: t, status: open, refutes: 0}\n"
            "candidates:\n"
            "  - {id: c1, direction: occupancy, status: pending}\n"
            "refuted: []\n")
        (self.card / "bench").mkdir(exist_ok=True)
        (self.card / "bench" / "ab.py").write_text("# stub\n", encoding="utf-8")
        d = self._data()
        self.assertTrue(d["bench_ready"])
        self.assertFalse(any("ab.py" in b["reason"] for b in d["blocked"]))



if __name__ == "__main__":
    unittest.main()