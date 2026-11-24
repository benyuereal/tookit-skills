#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scripts/state.py 的测试。

重点不是「看板能打印」，而是：
  - 阶段定义**确实从 validate.py 导入**（不是复制一份，否则必然分叉）；
  - 「没写 progress」「stage 非法」「blocked_by 非法」三类被分开计数，
    不混进阶段分布；
  - --check 能抓住过期的 STATE.md。
"""
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
STATE = REPO / "scripts/state.py"
VALIDATE = REPO / "Plugins/ops/op-atlas/validate.py"


def _load(path, modname):
    spec = importlib.util.spec_from_file_location(modname, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


class TestState(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state = _load(STATE, "_state_under_test")

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        self.cards = self.root / "cards"
        self.cards.mkdir()
        self.state_md = self.root / "STATE.md"

    def tearDown(self):
        self._td.cleanup()

    def _card(self, name, body):
        (self.cards / ("%s.yaml" % name)).write_text(body, encoding="utf-8")

    # ---- 阶段定义来源 ----
    def test_stages_imported_not_rewritten(self):
        # state.py 加载的 validate 模块确实来自那个文件。
        loaded_from = Path(self.state.VALIDATE.__file__).resolve()
        self.assertEqual(loaded_from, VALIDATE.resolve())
        # 三个对象是 validate 模块里的**同一个对象**，不是拷贝。
        self.assertIs(self.state.STAGES, self.state.VALIDATE.STAGES)
        self.assertIs(self.state.STAGE_NEEDS, self.state.VALIDATE.STAGE_NEEDS)
        self.assertIs(self.state.BLOCKED_BY, self.state.VALIDATE.BLOCKED_BY)
        # 内容也与重新加载的 validate.py 一致（防两文件漂移）。
        validate = _load(VALIDATE, "_validate_for_state_test")
        self.assertEqual(self.state.STAGES, validate.STAGES)
        self.assertEqual(self.state.STAGE_NEEDS, validate.STAGE_NEEDS)
        self.assertEqual(self.state.BLOCKED_BY, validate.BLOCKED_BY)
        # 源码里没有第二份阶段表：不出现 STAGES = ("S0", ...
        src = STATE.read_text(encoding="utf-8")
        self.assertNotIn('STAGES = ("S0"', src)
        self.assertNotIn("STAGE_NEEDS = {", src)

    # ---- 分桶计数 ----
    def test_counts_splits_progress_unadmitted_illegal(self):
        self._card("a", "id: a\nprogress:\n  stage: S1\n")
        self._card("b", "id: b\nprogress:\n  stage: S1\n")
        self._card("c", "id: c\nprogress:\n  stage: S6\n  blocked_by: no_baseline\n")
        self._card("d", "id: d\n")                       # 无 progress
        self._card("e", "id: e\nprogress:\n  stage: S9\n")  # 非法 stage
        self._card("f", "id: f\nprogress:\n  since: x\n")   # 有 progress 无 stage

        board = self.state.scan(self.cards)
        self.assertEqual(board["total"], 6)
        self.assertEqual(sorted(board["stages"]["S1"]), ["a", "b"])
        self.assertEqual(board["stages"]["S6"], ["c"])
        self.assertEqual(board["stages"]["S0"], [])       # 没写 progress 不默认算 S0
        self.assertEqual(sorted(board["unadmitted"]), ["d", "f"])
        self.assertEqual([n for n, _ in board["illegal"]], ["e"])
        self.assertEqual(board["blocked"], {"no_baseline": ["c"]})

    def test_blocked_grouped_and_bad_blocked_separate(self):
        self._card("a", "id: a\nprogress:\n  stage: S1\n  blocked_by: semantics\n")
        self._card("b", "id: b\nprogress:\n  stage: S1\n  blocked_by: semantics\n")
        self._card("c", "id: c\nprogress:\n  stage: S2\n  blocked_by: no_baseline\n")
        self._card("d", "id: d\nprogress:\n  stage: S2\n  blocked_by: made_up\n")

        board = self.state.scan(self.cards)
        self.assertEqual(board["blocked"],
                         {"semantics": ["a", "b"], "no_baseline": ["c"]})
        self.assertEqual([n for n, _ in board["bad_blocked"]], ["d"])

    # ---- 空目录 ----
    def test_empty_cards_dir_does_not_crash(self):
        board = self.state.scan(self.cards)
        self.assertEqual(board["total"], 0)
        self.assertTrue(all(v == [] for v in board["stages"].values()))
        text = self.state.render(board)
        self.assertIn("阶段分布", text)
        self.assertIn("未接入（0 张", text)

    def test_missing_cards_dir_does_not_crash(self):
        board = self.state.scan(self.root / "nope")
        self.assertEqual(board["total"], 0)

    # ---- --check ----
    def test_check_fails_when_state_stale(self):
        self._card("a", "id: a\nprogress:\n  stage: S1\n")
        self.state_md.write_text("stale\n", encoding="utf-8")
        rc = self.state.main(["--cards", str(self.cards),
                              "--state", str(self.state_md), "--check"])
        self.assertNotEqual(rc, 0)

    def test_check_fails_when_cards_changed_after_write(self):
        self._card("a", "id: a\nprogress:\n  stage: S1\n")
        self.assertEqual(self.state.main(["--cards", str(self.cards),
                                          "--state", str(self.state_md),
                                          "--write"]), 0)
        self._card("b", "id: b\nprogress:\n  stage: S2\n")
        rc = self.state.main(["--cards", str(self.cards),
                              "--state", str(self.state_md), "--check"])
        self.assertNotEqual(rc, 0)

    def test_check_passes_when_consistent(self):
        self._card("a", "id: a\nprogress:\n  stage: S1\n")
        self.assertEqual(self.state.main(["--cards", str(self.cards),
                                          "--state", str(self.state_md),
                                          "--write"]), 0)
        rc = self.state.main(["--cards", str(self.cards),
                              "--state", str(self.state_md), "--check"])
        self.assertEqual(rc, 0)

    def test_check_missing_file_is_nonzero(self):
        self._card("a", "id: a\nprogress:\n  stage: S1\n")
        rc = self.state.main(["--cards", str(self.cards),
                              "--state", str(self.state_md), "--check"])
        self.assertNotEqual(rc, 0)

    # ---- 输出 ----
    def test_json_shape(self):
        self._card("a", "id: a\nprogress:\n  stage: S2\n  blocked_by: no_counters\n")
        self._card("b", "id: b\n")
        buf = io.StringIO()
        with redirect_stdout(buf):
            self.assertEqual(self.state.main(["--cards", str(self.cards),
                                              "--json"]), 0)
        data = json.loads(buf.getvalue())
        self.assertEqual(data["total"], 2)
        self.assertEqual([s["stage"] for s in data["stages"]],
                         list(self.state.STAGES))
        self.assertEqual(data["blocked"]["by_reason"]["no_counters"]["cards"], ["a"])
        self.assertEqual(data["unadmitted"]["cards"], ["b"])

    def test_rendered_board_lists_card_names(self):
        self._card("alpha", "id: alpha\nprogress:\n  stage: S3\n")
        text = self.state.render(self.state.scan(self.cards))
        self.assertIn("S3", text)
        self.assertIn("- alpha", text)


if __name__ == "__main__":
    unittest.main()
