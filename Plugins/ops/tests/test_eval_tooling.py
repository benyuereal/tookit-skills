#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""评测工具链自己的测试：`scripts/run_evals.py`（静态）+ `scripts/skill_eval.py`（动态）。

断言的是**判定逻辑**，不调用模型、不连网：
  ① stream-json 解析：按顺序取 Skill 调用、跳过别的 tool_use、取 usage；
  ② 路由判分：true=第一个 Skill 必须是本 skill；false=全程不得出现本 skill；
  ③ 可发现性：带 evals 的 skill 必须能向上找到插件根（缺 `.claude-plugin/plugin.json`
     时 `claude -p --plugin-dir` 根本挂不上，动态评测会静默变「0 命中」）；
  ④ description 静态门：有 evals 的 skill 必须同时写「触发条件」与「不适用反例」
     （docs/STANDARDS.md），否则模型只能靠猜决定拉不拉。
"""
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))
import run_evals  # noqa: E402
import skill_eval  # noqa: E402


def skill_use(name):
    return json.dumps({"type": "assistant",
                       "message": {"content": [{"type": "tool_use", "name": "Skill",
                                                "input": {"skill": name}}]}})


class TestStreamParsing(unittest.TestCase):
    def test_calls_are_ordered_and_only_skill_tool_counts(self):
        stdout = "\n".join([
            "not json at all",
            skill_use("op-bench"),
            json.dumps({"type": "assistant", "message": {"content": [
                {"type": "tool_use", "name": "Read",
                 "input": {"file_path": "/x"}}]}}),
            skill_use("op-gate"),
            json.dumps({"type": "result", "usage": {"input_tokens": 100,
                                                    "output_tokens": 20}}),
        ])
        picked, tokens = skill_eval.parse_skill_calls(stdout)
        self.assertEqual(picked, ["op-bench", "op-gate"])
        self.assertEqual(tokens, 120)

    def test_no_usage_means_none(self):
        picked, tokens = skill_eval.parse_skill_calls(skill_use("x"))
        self.assertEqual(picked, ["x"])
        self.assertIsNone(tokens)

    def test_last_assistant_text_prefers_final_result(self):
        stdout = "\n".join([
            json.dumps({"type": "assistant", "message": {"content": [
                {"type": "text", "text": "中间过程"}]}}),
            json.dumps({"type": "result", "result": "最终回答"}),
        ])
        self.assertEqual(skill_eval.last_assistant_text(stdout), "最终回答")


class TestRouting(unittest.TestCase):
    def test_true_case_needs_this_skill_first(self):
        case = {"id": "x-t1", "expect": "要点"}
        self.assertTrue(skill_eval.score_routing("op-bench", case, ["op-bench"])[0])
        self.assertFalse(skill_eval.score_routing("op-bench", case, ["op-gate"])[0])
        self.assertFalse(skill_eval.score_routing("op-bench", case, ["op-gate", "op-bench"])[0],
                         "被别的 skill 抢先拉起也算没命中")
        self.assertFalse(skill_eval.score_routing("op-bench", case, [])[0])

    def test_false_case_must_not_fire_at_all(self):
        case = {"id": "x-f1"}
        self.assertTrue(skill_eval.score_routing("op-bench", case, ["op-gate"])[0])
        self.assertTrue(skill_eval.score_routing("op-bench", case, [])[0])
        self.assertFalse(skill_eval.score_routing("op-bench", case, ["op-gate", "op-bench"])[0])

    def test_plugin_namespace_is_stripped(self):
        """`--plugin-dir` 挂进来的 skill 叫 `ops:op-bench`，判分不能因此判错。"""
        case = {"id": "x-t1", "expect": "要点"}
        self.assertTrue(skill_eval.score_routing("op-bench", case, ["ops:op-bench"])[0])
        self.assertFalse(skill_eval.score_routing("op-bench", case,
                                                  ["dsh:x", "ops:op-bench"])[0])
        false_case = {"id": "x-f1"}
        self.assertFalse(skill_eval.score_routing("op-bench", false_case,
                                                   ["ops:op-gate", "ops:op-bench"])[0])


class TestRepeat(unittest.TestCase):
    """`--repeat N`：单次判分是概率事件，抖动本身要报出来，不能被一次通过洗白。"""

    def test_failure_reason_splits_three_cases(self):
        true_case = {"id": "x-t1", "expect": "要点"}
        self.assertEqual(skill_eval.failure_reason("op-bench", true_case, []), "未拉起")
        self.assertEqual(skill_eval.failure_reason("op-bench", true_case, ["op-gate"]), "拉错")
        self.assertEqual(skill_eval.failure_reason("op-bench", true_case, ["ops:op-bench"]), "")
        false_case = {"id": "x-f1"}
        self.assertEqual(skill_eval.failure_reason("op-bench", false_case, ["op-bench"]), "误拉")
        self.assertEqual(skill_eval.failure_reason("op-bench", false_case, ["op-gate"]), "")

    def _attempt(self, ok, reason="", sec=10.0, tokens=100):
        return dict(id="x-t1", skill="op-bench", kind="true", ok=ok, reason=reason,
                    detail="d", picked=[], sec=sec, tokens=tokens, prompt="p")

    def test_all_pass_is_pass(self):
        row = skill_eval.aggregate_attempts([self._attempt(True)] * 3)
        self.assertTrue(row["ok"])
        self.assertEqual((row["verdict"], row["passed"], row["attempts"]), ("PASS", 3, 3))
        self.assertEqual(row["tokens"], 300)

    def test_partial_pass_is_flaky_and_counts_as_failure(self):
        row = skill_eval.aggregate_attempts(
            [self._attempt(True), self._attempt(False, "未拉起"), self._attempt(False, "未拉起")])
        self.assertFalse(row["ok"], "抖动的用例不能算通过")
        self.assertEqual((row["verdict"], row["passed"]), ("FLAKY", 1))
        self.assertEqual(row["reason"], "未拉起")

    def test_all_fail_takes_majority_reason(self):
        row = skill_eval.aggregate_attempts(
            [self._attempt(False, "拉错"), self._attempt(False, "拉错"),
             self._attempt(False, "未拉起")])
        self.assertEqual(row["verdict"], "FAIL")
        self.assertEqual(row["reason"], "拉错")
        self.assertEqual(row["verdict"] + str(row["passed"]), "FAIL0")


class TestForbid(unittest.TestCase):
    """`forbid` = 路由对了之后判「有没有说错」，零额外 token（只读已有回答）。"""

    def test_wrong_claim_is_caught(self):
        case = {"forbid": ["re:TA_\\*[^。；\\n]{0,8}取平均"]}
        self.assertEqual(skill_eval.forbidden_hit(case, "TA_* 直接取平均就行"),
                         case["forbid"][0])

    def test_correct_negation_is_not_flagged(self):
        case = {"forbid": ["re:TA_\\*[^。；\\n]{0,8}取平均", "re:sshpass\\s+-p\\s+\\S+"]}
        self.assertEqual(skill_eval.forbidden_hit(case, "TA_* 不是取平均，要按 SE 求和"), "")
        self.assertEqual(skill_eval.forbidden_hit(case, "禁止 sshpass -p 123 ssh host"), "")
        self.assertEqual(skill_eval.forbidden_hit(case, "不要用 sshpass -p 传密码"), "")

    def test_literal_phrase_and_clean_answer(self):
        case = {"forbid": ["首次 launch 的数可以直接用"]}
        self.assertEqual(skill_eval.forbidden_hit(case, "首次 launch 的数可以直接用"), "首次 launch 的数可以直接用")
        self.assertEqual(skill_eval.forbidden_hit(case, "首次 launch 的数不能直接用，要预热"), "")
        self.assertEqual(skill_eval.forbidden_hit({"expect": "x"}, "随便聊聊"), "")

    def test_bad_regex_is_ignored_not_crashed(self):
        self.assertEqual(skill_eval.forbidden_hit({"forbid": ["re:("]}, "任意回答"), "")


class TestDiscoverability(unittest.TestCase):
    """带 evals 却挂不上插件 → 动态评测永远 0 命中（真实踩过：ops 缺 plugin.json）。"""

    def test_every_skill_with_evals_has_a_plugin_root(self):
        orphan = [d for d in skill_eval.iter_skill_dirs(str(REPO))
                  if skill_eval.load_case_file(d) and not skill_eval.plugin_dir_of(d)]
        self.assertEqual(orphan, [], "这些 skill 有 evals 但没有 .claude-plugin/plugin.json：%s"
                         % orphan)

    def test_ops_domain_is_runnable(self):
        targets = skill_eval.collect(str(REPO), set(), {"ops"})
        names = {name for name, _d, _p, _data in targets}
        self.assertIn("op-contract", names)
        self.assertIn("op-bench", names)

    def test_every_marketplace_plugin_has_plugin_json(self):
        """marketplace.json 声明了却缺 plugin.json → `--plugin-dir` 挂不上（ops 犯过）。"""
        mp = json.loads((REPO / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))
        missing = [p["name"] for p in mp["plugins"]
                   if not (REPO / p["source"] / ".claude-plugin" / "plugin.json").is_file()]
        self.assertEqual(missing, [], "这些插件缺 .claude-plugin/plugin.json：%s" % missing)

    def test_runner_uses_every_plugin(self):
        dirs = skill_eval.all_plugin_dirs(str(REPO))
        self.assertGreaterEqual(len(dirs), 8)
        self.assertIn(str(REPO / "Plugins" / "ops"), dirs)


class TestDescriptionLint(unittest.TestCase):
    def test_gaps_are_detected_semantically(self):
        self.assertEqual(run_evals.description_gaps(
            "查镜像。触发场景：找某个 DTK 版本。不适用：怎么起容器走 dcu-docker。"), [])
        self.assertEqual(run_evals.description_gaps("Operate DCU environments inside Docker."),
                         ["触发条件", "不适用反例"])
        self.assertEqual(run_evals.description_gaps("触发场景：A。"), ["不适用反例"])
        self.assertEqual(run_evals.description_gaps(None),
                         ["触发条件", "不适用反例"])

    def test_every_evaluated_skill_describes_its_boundary(self):
        bad = []
        for d in skill_eval.iter_skill_dirs(str(REPO)):
            if not skill_eval.load_case_file(d):
                continue
            desc = run_evals.read_description(str(Path(d) / "SKILL.md"))
            gaps = run_evals.description_gaps(desc)
            if gaps:
                bad.append("%s: 缺 %s" % (Path(d).name, "/".join(gaps)))
        self.assertEqual(bad, [], "\n".join(bad))


class TestCLI(unittest.TestCase):
    """参数层的早退路径：不匹配的 --case 必须退 2，别悄悄「0 例全过」。"""

    def _run(self, argv):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            return skill_eval.main(["--root", str(REPO)] + argv), buf.getvalue()

    def test_unknown_case_is_an_argument_error(self):
        rc, _out = self._run(["--case", "no-such-case"])
        self.assertEqual(rc, 2)

    def test_list_never_calls_the_model(self):
        rc, out = self._run(["--list"])
        self.assertEqual(rc, 0)
        self.assertIn("op-contract", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
