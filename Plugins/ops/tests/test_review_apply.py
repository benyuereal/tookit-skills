#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""review_apply.py 的反向测试 —— **人工无提权**。

业主裁定（`/root/atlas-handoff/WORKFLOW-UPGRADE-PLAN.md` §3.4）：
> 人工批准只能把状态标成 blocked 或待复核，绝不能把 E41–E43 的失败改成通过。

所以本文件的每条用例都是「故意伪造一次提权，确认它被拒」，而不是
「合法路径能跑通」。合法路径只作为对照组存在——否则判据太宽也会全绿。
"""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]


def _load():
    spec = importlib.util.spec_from_file_location(
        "_review_apply", str(REPO / "scripts/review_apply.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CARD_S6 = {
    "id": "rmsnorm-fused",
    "title": "t",
    "progress": {"stage": "S6", "since": "2026-11-24T02:05",
                 "blocked_by": None, "last_verdict": "adopt", "rounds": 3},
    "last_verdict": {"status": "measured", "rounds": 3,
                     "verdict": "adopt", "resources": {"vgpr": 80}},
    "gates_note": "E41 pass / E42 pass / E43 pass",
}


def _review(**kw):
    base = {"id": "2026-11-24T02:10:11/rmsnorm-fused/S6",
            "reviewer": "owner", "action": "downgrade", "to": "S4",
            "reason": "resources.status=partial", "ts": "2026-11-24T09:00:00"}
    base.update(kw)
    return base


class TestReviewApply(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cards = self.root / "cards"
        self.cards.mkdir()
        self.reviews = self.root / "REVIEWS.jsonl"
        self.card_path = self.cards / "rmsnorm-fused.yaml"

    def tearDown(self):
        self.tmp.cleanup()

    def write_card(self, card=None):
        self.card_path.write_text(
            yaml.dump(card or CARD_S6, allow_unicode=True, sort_keys=False),
            encoding="utf-8")

    def read_card(self):
        return yaml.safe_load(self.card_path.read_text(encoding="utf-8"))

    def write_reviews(self, rows):
        self.reviews.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
            encoding="utf-8")

    def apply(self, rows, dry_run=False):
        self.write_reviews(rows)
        return self.m.apply_reviews(self.reviews, self.cards, dry_run=dry_run)

    # ---- 1. 合法降级 S6 -> S4 ----
    def test_downgrade_s6_to_s4_succeeds(self):
        self.write_card()
        rep = self.apply([_review(action="downgrade", to="S4")])
        self.assertTrue(rep["ok"], rep)
        card = self.read_card()
        self.assertEqual(card["progress"]["stage"], "S4")

    # ---- 2. 提权 S4 -> S7 必须被拒 ----
    def test_downgrade_s4_to_s7_rejected(self):
        card = dict(CARD_S6, progress=dict(CARD_S6["progress"], stage="S4"))
        self.write_card(card)
        rep = self.apply([_review(action="downgrade", to="S7")])
        self.assertFalse(rep["ok"], "把阶段往上调必须失败")
        self.assertTrue(any(self.m.NO_ESCALATION in msg
                            for _, _, msg in rep["rejected"]), rep)
        self.assertEqual(self.read_card()["progress"]["stage"], "S4",
                         "被拒的行不许改卡")

    def test_downgrade_same_stage_not_escalation(self):
        """S4->S4 是幂等，不是提权，不该被判「人工无提权」。"""
        card = dict(CARD_S6, progress=dict(CARD_S6["progress"], stage="S4"))
        self.write_card(card)
        rep = self.apply([_review(action="downgrade", to="S4")])
        self.assertTrue(rep["ok"], rep)

    # ---- 3. 试图清除 gate 失败 -> 拒绝 ----
    def test_clear_gate_failure_rejected(self):
        self.write_card()
        before = self.read_card()
        for key in ("gates", "gate", "clear_gates", "set_gates",
                    "E42", "override_gates", "force"):
            with self.subTest(key=key):
                self.write_card()
                rep = self.apply([_review(action="downgrade", to="S4",
                                          **{key: {"E42": "pass"}})])
                self.assertFalse(rep["ok"], "带 %s 的行必须被拒" % key)
                self.assertTrue(any(self.m.NO_ESCALATION in m
                                    for _, _, m in rep["rejected"]), rep)
                self.assertEqual(self.read_card(), before,
                                 "带 %s 的行不许改卡" % key)

    # ---- 4. ack 不改任何字段 ----
    def test_ack_changes_nothing(self):
        self.write_card()
        before = self.read_card()
        rep = self.apply([_review(action="ack", to=None, reason="认可")])
        self.assertTrue(rep["ok"], rep)
        self.assertEqual(self.read_card(), before, "ack 必须一个字段都不改")

    def test_ack_with_to_rejected(self):
        self.write_card()
        rep = self.apply([_review(action="ack", to="S4")])
        self.assertFalse(rep["ok"], "ack 带 to 是想顺手改，必须拒")

    # ---- 5. suspect 只加标记 ----
    def test_suspect_only_sets_flag(self):
        self.write_card()
        before = self.read_card()
        rep = self.apply([_review(action="suspect", to=None,
                                  reason="VGPR 可疑")])
        self.assertTrue(rep["ok"], rep)
        after = self.read_card()
        self.assertIs(after["suspect"], True)
        self.assertEqual(after["progress"], before["progress"],
                         "suspect 不许动 progress")
        self.assertEqual(after["last_verdict"], before["last_verdict"],
                         "suspect 不许动门禁结论")

    def test_suspect_with_to_rejected(self):
        self.write_card()
        rep = self.apply([_review(action="suspect", to="S5",
                                  reason="x")])
        self.assertFalse(rep["ok"])

    # ---- 6. blocked_by 非法值 -> 拒绝 ----
    def test_illegal_blocked_by_rejected(self):
        self.write_card()
        rep = self.apply([_review(action="downgrade", to="blocked",
                                  blocked_by="whatever")])
        self.assertFalse(rep["ok"], "blocked_by 只接受枚举值")
        self.assertEqual(self.read_card()["progress"]["stage"], "S6")

    def test_blocked_by_must_match_validate_enum(self):
        """脚本的 BLOCKED_BY 必须与 validate.py 完全一致。"""
        spec = importlib.util.spec_from_file_location(
            "_val_rb", str(REPO / "Plugins/ops/op-atlas/validate.py"))
        val = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(val)
        self.assertEqual(self.m.BLOCKED_BY, val.BLOCKED_BY)

    def test_valid_blocked_by_applied(self):
        self.write_card()
        rep = self.apply([_review(action="downgrade", to="blocked",
                                  blocked_by="no_resources")])
        self.assertTrue(rep["ok"], rep)
        p = self.read_card()["progress"]
        self.assertEqual(p["blocked_by"], "no_resources")
        self.assertEqual(p["stage"], "S6", "只设 blocked_by 不该改 stage")

    # ---- 7. REVIEWS 旧行不被改写（跑两遍内容不变） ----
    def test_reviews_file_untouched_across_two_runs(self):
        self.write_card()
        rows = [_review(action="downgrade", to="S4")]
        self.write_reviews(rows)
        original = self.reviews.read_bytes()
        self.m.apply_reviews(self.reviews, self.cards)
        self.assertEqual(self.reviews.read_bytes(), original,
                         "第一次跑就改写了 REVIEWS")
        self.m.apply_reviews(self.reviews, self.cards)
        self.assertEqual(self.reviews.read_bytes(), original,
                         "跑两遍后 REVIEWS 内容必须不变（append-only）")

    # ---- 8. 空 / 不存在的 REVIEWS -> 正常退出 ----
    def test_empty_reviews_ok(self):
        self.reviews.write_text("", encoding="utf-8")
        rep = self.m.apply_reviews(self.reviews, self.cards)
        self.assertTrue(rep["ok"])
        self.assertEqual(self.m.main(["--reviews", str(self.reviews),
                                      "--cards", str(self.cards)]), 0)

    def test_missing_reviews_ok(self):
        rep = self.m.apply_reviews(self.reviews, self.cards)
        self.assertTrue(rep["ok"])
        self.assertEqual(self.m.main(["--reviews", str(self.reviews),
                                      "--cards", str(self.cards)]), 0)
        self.assertFalse(self.reviews.exists(), "只读脚本不许凭空创建 REVIEWS")

    # ---- 附加：CLI 退出码与路径安全 ----
    def test_candidate_exit_nonzero_on_escalation(self):
        card = dict(CARD_S6, progress=dict(CARD_S6["progress"], stage="S4"))
        self.write_card(card)
        self.write_reviews([_review(action="downgrade", to="S7")])
        rc = self.m.main(["--reviews", str(self.reviews),
                          "--cards", str(self.cards)])
        self.assertNotEqual(rc, 0, "有拒行时退出码必须非 0")

    def test_dry_run_does_not_write(self):
        self.write_card()
        before = self.card_path.read_bytes()
        rep = self.apply([_review(action="downgrade", to="S4")], dry_run=True)
        self.assertTrue(rep["ok"])
        self.assertEqual(self.card_path.read_bytes(), before)

    def test_illegal_action_rejected(self):
        self.write_card()
        rep = self.apply([_review(action="escalate", to="S7")])
        self.assertFalse(rep["ok"])

    def test_unknown_action_key_rejected(self):
        self.write_card()
        rep = self.apply([_review(action="downgrade", to="S4", magic=1)])
        self.assertFalse(rep["ok"])
        self.assertTrue(any(self.m.NO_ESCALATION in m
                            for _, _, m in rep["rejected"]))


if __name__ == "__main__":
    unittest.main()
