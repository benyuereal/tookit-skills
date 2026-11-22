#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""场景映射（scenarios.yaml）守卫 —— S1 的机检部分。

为什么需要这层：surfaces.yaml 的 11 条攻击面是**平铺**的，它回答
「有哪些方向」，没回答「我这个场景该先试哪几条」。没有场景层，
agent 会把 11 条全试一遍，或者凭直觉挑一条——而那条往往不是这个场景的杠杆。

五条断言：
  1. 场景 ≥4 个，id 唯一且 kebab-case
  2. 每个场景挂 ≥3 条 surface，且都是 surfaces.yaml 里真实存在的 id
  3. 每个场景有优先序（surfaces 是列表，顺序即优先序）
  4. 每个场景有 detect / dead_ends / bound
  5. 场景之间不要求互斥，但 boundaries 必须给冲突规则
"""
import pathlib
import re
import unittest

REPO = pathlib.Path(__file__).resolve().parents[3]
REF = REPO / "Plugins/ops/skills/op-optimize/references"
SCEN = REF / "scenarios.yaml"
SURF = REF / "surfaces.yaml"

KEBAB = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


def _load(p):
    try:
        import yaml
    except ImportError:                       # pragma: no cover
        return None
    return yaml.safe_load(p.read_text(encoding="utf-8"))


class TestScenarios(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = _load(SCEN)
        if cls.data is None:
            raise unittest.SkipTest("PyYAML 未安装")
        cls.surfaces = {s["id"] for s in (_load(SURF) or [])}

    def test_at_least_four_scenarios(self):
        sc = self.data.get("scenarios") or []
        self.assertGreaterEqual(len(sc), 4, "场景数 %d < 4" % len(sc))
        ids = [s.get("id") for s in sc]
        self.assertEqual(len(ids), len(set(ids)), "场景 id 有重复")
        bad = [i for i in ids if not KEBAB.match(str(i))]
        self.assertEqual(bad, [], "id 不是 kebab-case：%s" % bad)

    def test_each_scenario_has_three_plus_real_surfaces(self):
        bad = []
        for s in self.data.get("scenarios") or []:
            ss = s.get("surfaces") or []
            if len(ss) < 3:
                bad.append("%s 只挂 %d 条 surface（要求 ≥3）" % (s.get("id"), len(ss)))
                continue
            for item in ss:
                sid = item.get("id") if isinstance(item, dict) else item
                if sid not in self.surfaces:
                    bad.append("%s 挂了不存在的 surface %r" % (s.get("id"), sid))
        self.assertEqual(bad, [], "\n".join(bad))

    def test_each_surface_entry_explains_why_and_how_to_check(self):
        """优先序不能只是列表——每条要写「为什么在这个场景优先」与「怎么验」。"""
        bad = []
        for s in self.data.get("scenarios") or []:
            for item in s.get("surfaces") or []:
                if not isinstance(item, dict):
                    bad.append("%s: surface 必须是带 why/check 的 mapping" % s.get("id"))
                    continue
                if not item.get("why"):
                    bad.append("%s/%s 缺 why（凭什么排这个位置）"
                               % (s.get("id"), item.get("id")))
                if not item.get("check"):
                    bad.append("%s/%s 缺 check（怎么验它成立）"
                               % (s.get("id"), item.get("id")))
        self.assertEqual(bad, [], "\n".join(bad))

    def test_each_scenario_has_detect_and_dead_ends(self):
        bad = []
        for s in self.data.get("scenarios") or []:
            for k in ("statement", "detect", "bound", "dead_ends"):
                if not s.get(k):
                    bad.append("%s 缺 %s" % (s.get("id"), k))
            if s.get("dead_ends") and len(s["dead_ends"]) < 2:
                bad.append("%s 的 dead_ends 少于 2 条（场景级死路通常不止一条）"
                           % s.get("id"))
        self.assertEqual(bad, [], "\n".join(bad))

    def test_boundaries_document_conflicts(self):
        """场景可能同时匹配；必须有冲突规则，否则攻击面优先序会用反。"""
        bs = self.data.get("boundaries") or []
        self.assertGreaterEqual(len(bs), 2, "boundaries 至少 2 条冲突规则")
        bad = [b for b in bs if not b.get("rule") or not b.get("why")]
        self.assertEqual(bad, [], "boundaries 每条都要有 rule 与 why")


if __name__ == "__main__":
    unittest.main()
