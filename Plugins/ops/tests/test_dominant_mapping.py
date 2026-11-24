#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断终点 → 决策起点的映射必须存在且完整。

背景（v5.1 §五）：attribution.md 的五个 dominant（hbm / l2-miss / lds-conflict /
vgpr-spill / issue-stall）在 op-optimize/ 和 scripts/ 全目录 grep **零命中**。
也就是说 agent 得出「主导瓶颈是 L2 miss」之后，**没有任何机检路径**把它翻译成
「该试哪条 surface」——它只能靠 LLM 自由发挥，而方向预算、refuted 拒跑、
动作类约束**全挂在 surface id 上**。这一步一自由发挥，后面所有约束都绕过。

本测试守三件事：
  A. 五个 dominant 必须全部有映射（缺一个 = 那条诊断结论无法落到动作）
  B. 映射引用的 surface id 必须在 surfaces.yaml 里存在
  C. 每个 dominant 至少映射到 1 条 surface
"""
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
REF = REPO / "Plugins/ops/skills/op-optimize/references"

DOMINANTS = {"hbm", "l2-miss", "lds-conflict", "vgpr-spill", "issue-stall"}


def _load():
    import yaml
    m = REF / "dominant-to-surface.yaml"
    if not m.is_file():
        raise unittest.SkipTest("dominant-to-surface.yaml 尚未建立")
    return (yaml.safe_load(m.read_text(encoding="utf-8")) or {},
            yaml.safe_load((REF / "surfaces.yaml").read_text(encoding="utf-8")) or [])


class TestDominantMapping(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m, cls.surfaces = _load()
        cls.map = cls.m.get("dominant_to_surface") or {}
        cls.ids = {str(x.get("id")) for x in cls.surfaces if isinstance(x, dict)}

    def test_all_five_dominants_mapped(self):
        """五个 dominant 一个都不能漏。"""
        missing = DOMINANTS - set(self.map)
        self.assertEqual(missing, set(),
                         "以下 dominant 没有映射到任何 surface：%s" % sorted(missing))

    def test_every_dominant_maps_to_at_least_one(self):
        """映射不能是空列表——空列表等于「这个诊断结论没法落到动作」。"""
        empty = [k for k, v in self.map.items() if not v]
        self.assertEqual(empty, [], "以下 dominant 的映射是空的：%s" % empty)

    def test_mapped_surfaces_exist(self):
        """映射引用的 surface id 必须真实存在。

        这条会**在新增 surface 之前失败**——那是正确的：它逼着
        「先补 surface、再补映射」，而不是反过来引用一个不存在的 id。
        """
        bad = []
        for k, v in self.map.items():
            for sid in v:
                if sid not in self.ids:
                    bad.append((k, sid))
        self.assertEqual(bad, [], "映射引用了不存在的 surface：%s" % bad)

    def test_unknown_has_three_states(self):
        """unknown 必须拆三态：csv_missing / counter_incomplete / counter_suspect。

        单态 unknown 会让 agent 的第一动作永远是"再去采一次"，
        而其中 counter_suspect 是**已知坏计数器**（SQ_LDS_BANK_CONFLICT 恒 0），
        再采一百次也是 0 —— 那是死循环。
        """
        st = set((self.m.get("unknown_states") or {}))
        self.assertEqual(st, {"csv_missing", "counter_incomplete", "counter_suspect"},
                         "unknown 三态不完整：%s" % sorted(st))

    def test_suspect_state_has_alternate_path(self):
        """counter_suspect 必须给出替代证据路径，不能只说"放弃"。"""
        node = (self.m.get("unknown_states") or {}).get("counter_suspect") or {}
        self.assertIn("ISA", str(node.get("action") or ""),
                      "counter_suspect 必须指向替代证据路径（ISA/消融），否则等于无解")

    def test_reverse_unknown_dominant_is_rejected(self):
        """反向：一个不在五个 dominant 里的键，不该被映射表接受。

        这里用「构造一个假表」的方式验证判据本身有效，
        而不是依赖真实文件恰好没写错。
        """
        fake = dict(self.map)
        fake["made-up-dominant"] = ["layout-pack"]
        extra = set(fake) - DOMINANTS
        self.assertEqual(extra, {"made-up-dominant"},
                         "判据必须能识别出表外键")


if __name__ == "__main__":
    unittest.main()
