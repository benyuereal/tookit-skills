#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""选型器（atlas_query select）的 5 个反例测试 —— P1 验收线。

业主：「如果选型器不可信，前面所有 schema 都只是文档。」

这五个反例各对应一次真实踩过的坑：

1. **流量不同（201MB vs 268MB）→ 拒绝同榜**
   2026-11-23 我们写 1 份输出（201MB），lightop 写 2 份（268MB），
   得出「快 1.18×」——少干 1/3 的活。
2. **差 1µs 但 MAD 也 1µs → 判并列**
   不许说「A 比 B 好」。这是 3×MAD 判据的直接应用。
3. **只有无 caliber 的历史数字 → 拒绝推荐**
   0.51 / 887 那类「有数但追不到来源」的，不许进排名。
4. **correctness_only 的实现最快 → 不许选中**
   快不等于可用。torch 三段式最快也不能当性能基准。
5. **同一后端跨两 regime → 分档选，不许合并**
   同一份数据在两种口径下测出两个数，不能放一张表。
"""
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "scripts/atlas_query.py"


class TestAtlasQuerySelftest(unittest.TestCase):
    def test_five_counterexamples_pass(self):
        r = subprocess.run([sys.executable, "-I", str(TOOL), "--selftest"],
                           capture_output=True, text=True)
        self.assertIn("selftest PASS", r.stdout, r.stdout + r.stderr)
        # 五个反例都要出现「ok」
        for i in (1, 2, 3, 4, 5):
            self.assertIn("[ok] 反例%d" % i, r.stdout,
                          "反例 %d 没有跑或没通过" % i)

    def test_where_lists_real_entries(self):
        r = subprocess.run([sys.executable, "-I", str(TOOL), "where", "rmsnorm"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("共", r.stdout)

    def test_select_without_data_refuses_to_rank(self):
        """真仓里 perf/ 还是空的 ⇒ select 必须拒绝排名并给 measure_cmd。"""
        r = subprocess.run([sys.executable, "-I", str(TOOL), "select", "rmsnorm",
                            "--shape", "prefill-8k", "--json"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            self.skipTest("select 无匹配卡（perf/ 尚空），跳过")
        import json
        d = json.loads(r.stdout)
        if d.get("chosen") is None:
            self.assertIn(d.get("confidence"), ("none", "low"))
            self.assertTrue(d.get("measure_cmd"),
                            "无数据时必须给 measure_cmd，而不是编一个推荐")


class TestSelectLogicInline(unittest.TestCase):
    """直接把 select() 当函数测，覆盖五条规则。"""

    @classmethod
    def setUpClass(cls):
        import importlib.util
        spec = importlib.util.spec_from_file_location("_aq", str(TOOL))
        cls.aq = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.aq)

    def _card(self, cid, ips):
        return (cid, {"id": cid,
                      "variants": [{"import_path": ip, "role": role,
                                    "semantics_match": sm}
                                   for ip, role, sm in ips]})

    def _perf(self, ip, us, mad=0.5, n=7, tp=None):
        return {"variant": ip, "median_us": us,
                "caliber": {"what": "op_call", "alloc_in_timing": False,
                            "state": "hot", "input_strategy": "preclone",
                            "timer": "hip_event", "rounds": 7,
                            "statistic": "median", "n": n, "mad": mad},
                "shape": {"M": 8192, "dtype": "bfloat16", "name": "prefill-8k"},
                "traffic_passes": tp}

    def test_traffic_mismatch_refused(self):
        cards = [self._card("x", [("a.x", "competitor", "exact"),
                                  ("b.y", "competitor", "exact")])]
        perfs = [self._perf("a.x", 200.0, tp={"reads": 2, "writes": 1}),
                 self._perf("b.y", 240.0, tp={"reads": 2, "writes": 2})]
        d = self.aq.select("x", "prefill-8k", "bfloat16", cards=cards, perfs=perfs)
        self.assertIsNone(d["chosen"])
        self.assertIn("不可比", d["note"])

    def test_tie_detected(self):
        cards = [self._card("x", [("c.p", "competitor", "exact"),
                                  ("d.q", "competitor", "exact")])]
        perfs = [self._perf("c.p", 100.0, mad=1.0), self._perf("d.q", 101.0, mad=1.0)]
        d = self.aq.select("x", "prefill-8k", "bfloat16", cards=cards, perfs=perfs)
        self.assertTrue(d["tie"])
        self.assertIn("并列", d["note"])

    def test_correctness_only_never_selected(self):
        cards = [self._card("x", [("fast.torch", "correctness_only", "exact"),
                                  ("slow.comp", "competitor", "exact")])]
        perfs = [self._perf("fast.torch", 10.0), self._perf("slow.comp", 100.0)]
        d = self.aq.select("x", "prefill-8k", "bfloat16", cards=cards, perfs=perfs)
        self.assertEqual(d["chosen"], "slow.comp")

    def test_unverified_never_selected(self):
        cards = [self._card("x", [("bad.u", "unverified", "exact"),
                                  ("good.c", "competitor", "exact")])]
        perfs = [self._perf("bad.u", 1.0), self._perf("good.c", 100.0)]
        d = self.aq.select("x", "prefill-8k", "bfloat16", cards=cards, perfs=perfs)
        self.assertEqual(d["chosen"], "good.c")

    def test_partial_semantics_refused_without_flag(self):
        cards = [self._card("x", [("part.p", "competitor", "partial")])]
        perfs = [self._perf("part.p", 50.0)]
        d = self.aq.select("x", "prefill-8k", "bfloat16", cards=cards, perfs=perfs)
        self.assertIsNone(d["chosen"])
        d2 = self.aq.select("x", "prefill-8k", "bfloat16", allow_partial=True,
                            cards=cards, perfs=perfs)
        self.assertEqual(d2["chosen"], "part.p")

    def test_caliber_mismatch_refused(self):
        cards = [self._card("x", [("m.n", "competitor", "exact")])]
        p1 = self._perf("m.n", 100.0, tp={"reads": 2, "writes": 2})
        p2 = self._perf("m.n", 50.0, tp={"reads": 2, "writes": 2})
        p2["caliber"] = dict(p2["caliber"], input_strategy="reuse")
        d = self.aq.select("x", "prefill-8k", "bfloat16", cards=cards, perfs=[p1, p2])
        self.assertIsNone(d["chosen"])
        self.assertIn("无法直接比", d["note"])


if __name__ == "__main__":
    unittest.main()
