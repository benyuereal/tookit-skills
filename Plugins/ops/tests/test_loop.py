#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""op-optimize/loop.py 单测：队列推进 + 失败分类/预算 + 机器判定（纯逻辑）。

被测脚本：<repo>/Plugins/ops/skills/op-optimize/loop.py
GPU 无关：判定（decide / _significant）、YAML 文本改写（_item_block / _set_field /
_drop_refuted / _to_refuted）、失败预算（budget_problems / impl-broken /
direction-refuted）、落盘记录、队列/状态/报告渲染，以及所有「拒绝跑」路径
（拒绝在起子进程之前就返回，不会真的调 bench/ab.py）。

candidates.yaml 的读写走 loop.py 内部的 `import yaml`；未装 PyYAML 的环境下这部分用例
显式 skip（纯文本与纯判定的用例仍然全部执行）。
"""
import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import yaml  # noqa: F401
    HAVE_YAML = True
except Exception:  # pragma: no cover - 取决于环境
    HAVE_YAML = False

REPO = Path(__file__).resolve().parents[3]
# 直接加载 engine 版。经 shim 加载会丢可变全局的共享
# （测试改 loop.VERIFY_TIMEOUT_S 时 engine 读不到），这类测试必须直连实现。
LOOP_PY = REPO / "Plugins" / "ops" / "engine" / "loop.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


loop = _load("op_loop_under_test", LOOP_PY)

# 测试**不许**往真实 memory/ 写。实测一次全量 pytest 灌了 147 条 tmpXXXX 假记录，
# 而 memory/ledger.jsonl 是跨卡检索的索引 —— 被污染后检索结果全是测试数据。
loop.MEMORY_WRITE = False

CARD_YAML = """\
defaults:
  bench: bench/ab.py
  s: 128
  skv: [32768, 131072]

directions:
  - id: split-k
    statement: "把 4 路 KV 切分合并，砍掉重复载入"
    status: open
    refutes: 0

  - id: raise-heads
    statement: "抬 heads/wave 稀释每 wave 重读 KV"
    status: open
    refutes: 0

  - id: shorten-max-chain
    statement: "缩短 online-softmax 的 max 链"
    status: open
    refutes: 0

candidates:
  - id: maxn32
    direction: shorten-max-chain
    parent_surface: cross-lane-fold
    status: pending
    priority: 1
    action_class: pipeline-depth
    hypothesis: "running max 每 32 块更新一次"
    expect: "-11%"
    evidence: "ledger 30"

  - id: kv4
    direction: split-k
    priority: 2
    action_class: reduce-bytes
    hypothesis: "kv 切 4 份"
    expect: "-3%"

refuted:
  - id: merge
    kind: impl
    direction: split-k
    why: "慢 2.12×，VGPR 256 / spill 30：4 路切分是寄存器生命周期分段，合并后生命周期全程重叠。"
"""


def _record(tag="maxn32", skv=32768, cand_ms=0.69, base_ms=0.78, decision="ACCEPTED_BITEXACT"):
    return {
        "ts": "2026-10-28T10:00:00", "card": "tmp", "tag": tag, "skv": skv,
        "shape": {"s": 128}, "base_ms": base_ms, "base_mad": 0.004,
        "fm_ms": 0.40, "fm_ratio": 0.58, "cand_ms": cand_ms, "mad": 0.003,
        "delta_pct": -11.7, "ratio_vs_base": cand_ms / base_ms,
        "maxdiff": 0.0, "rel": 0.0, "acc": {}, "tolerance": 2e-3,
        "decision": decision, "why": "显著", "forced": False,
        # P1 之后 verify/report 也要求证据种类；夹具给独立核验
        "evidence_kind": "vendor_library",
    }


class TestDecide(unittest.TestCase):
    """判定由脚本算，不由模型说——四档 + 精度两层的逻辑都在这里。"""

    base = {"median_ms": 10.0, "mad_ms": 0.2}
    tol = 2e-3

    def _cand(self, ms=8.0, **over):
        c = {"ms": ms, "mad": 0.2, "delta_pct": (ms / 10.0 - 1.0) * 100.0, "maxdiff": 0.0}
        c.update(over)
        return c

    def test_bit_exact_accept(self):
        d, why = loop.decide(self.base, self._cand(), None, None, self.tol, "1", 1.5)
        self.assertEqual(d, "ACCEPTED_BITEXACT", why)
        self.assertIn("逐位一致", why)

    def test_non_bit_exact_needs_e2e(self):
        d, why = loop.decide(self.base, self._cand(maxdiff=1.2e-3), None, None, self.tol, "1", 1.5)
        self.assertEqual(d, "ACCEPTED_NUMERICS", why)
        self.assertIn("端到端", why)

    def test_inconclusive_when_within_noise(self):
        d, why = loop.decide(self.base, self._cand(ms=9.9), None, None, self.tol, "1", 1.5)
        self.assertEqual(d, "INCONCLUSIVE", why)
        self.assertIn("噪声内", why)

    def test_slower_is_rejected_no_gain(self):
        d, why = loop.decide(self.base, self._cand(ms=11.0), None, None, self.tol, "1", 1.5)
        self.assertEqual(d, "REJECTED_NO_GAIN", why)

    def test_error_candidate_short_circuits(self):
        d, why = loop.decide(self.base, {"error": "kernel 编译失败"}, None, None, self.tol, "1", 1.5)
        self.assertEqual(d, "ERROR")
        self.assertIn("编译失败", why)

    def test_primary_qscale_accuracy_is_a_hard_gate(self):
        acc = {"1": {"max_abs": 5e-3, "nonfinite": 0}}
        d, why = loop.decide(self.base, self._cand(), None, acc, self.tol, "1", 1.5)
        self.assertEqual(d, "REJECTED_ACCURACY", why)
        self.assertIn("容差", why)

    def test_nonfinite_at_any_qscale_rejects(self):
        acc = {"1": {"max_abs": 1e-4, "nonfinite": 0}, "10": {"max_abs": 1e-3, "nonfinite": 2}}
        d, why = loop.decide(self.base, self._cand(), None, acc, self.tol, "1", 1.5)
        self.assertEqual(d, "REJECTED_ACCURACY", why)
        self.assertIn("nonfinite", why)

    def test_absolute_tolerance_is_not_applied_to_stress_qscale(self):
        # base 自己在 qscale=10 就有 1.5e-2；压力档只用绝对容差会把正确改动一起否掉
        acc = {"1": {"max_abs": 1e-4, "nonfinite": 0},
               "10": {"max_abs": 1.5e-2, "nonfinite": 0}}
        d, why = loop.decide(self.base, self._cand(), None, acc, self.tol, "1", 1.5)
        self.assertEqual(d, "ACCEPTED_BITEXACT", why)

    def test_stress_qscale_relative_regression_only_warns(self):
        acc_base = {"10": {"rel": 1.0e-3}}
        acc_cand = {"1": {"max_abs": 1e-4, "nonfinite": 0},
                    "10": {"max_abs": 1e-4, "nonfinite": 0, "rel": 3.0e-3}}
        d, why = loop.decide(self.base, self._cand(), acc_base, acc_cand, self.tol, "1", 1.5)
        self.assertEqual(d, "ACCEPTED_BITEXACT", why)
        self.assertIn("退化", why)
        self.assertIn("端到端", why)

    def test_stress_regression_under_multiplier_does_not_warn(self):
        acc_base = {"10": {"rel": 1.0e-3}}
        acc_cand = {"10": {"max_abs": 1e-4, "nonfinite": 0, "rel": 1.2e-3}}
        d, why = loop.decide(self.base, self._cand(), acc_base, acc_cand, 2e-3, "1", 1.5)
        self.assertEqual(d, "ACCEPTED_BITEXACT", why)
        self.assertNotIn("退化", why)

    def test_significant_boundary_matches_noise_gate(self):
        ok, ratio, why = loop._significant({"median_ms": 10.0, "mad_ms": 0.2},
                                            {"median_ms": 9.4, "mad_ms": 0.2})
        self.assertFalse(ok, why)
        self.assertAlmostEqual(ratio, 0.94)
        ok2, _, _ = loop._significant({"median_ms": 10.0, "mad_ms": 0.2},
                                      {"median_ms": 9.3, "mad_ms": 0.2})
        self.assertTrue(ok2)


class TestTextTransforms(unittest.TestCase):
    """YAML 文本改写的核心，纯字符串，不需要 PyYAML。"""

    def test_cand_block_bounds(self):
        start, end = loop._cand_block(CARD_YAML, "kv4")
        block = CARD_YAML[start:end]
        self.assertTrue(block.startswith("  - id: kv4"), repr(block))
        self.assertNotIn("refuted", block)

    def test_cand_block_ignores_refuted_same_id(self):
        self.assertIsNone(loop._cand_block(CARD_YAML, "merge"))
        # refuted 里的 merge 没有被 candidates 段误命中
        self.assertIn("  - id: merge", CARD_YAML)

    def test_item_block_is_section_scoped(self):
        # 三段都有自己的 `- id:` 列表：只在被点名的段里找
        self.assertIsNotNone(loop._item_block(CARD_YAML, "directions", "raise-heads"))
        self.assertIsNone(loop._item_block(CARD_YAML, "candidates", "raise-heads"))
        self.assertIsNotNone(loop._item_block(CARD_YAML, "candidates", "kv4"))

    def test_cand_block_absent(self):
        self.assertIsNone(loop._cand_block(CARD_YAML, "nothere"))

    def test_section_span_stops_at_next_top_level_key(self):
        s, e = loop._section_span(CARD_YAML, "directions")
        seg = CARD_YAML[s:e]
        self.assertTrue(seg.startswith("directions:"))
        self.assertIn("raise-heads", seg)
        self.assertNotIn("candidates:", seg)

    def test_drop_refuted_is_idempotent_on_text(self):
        once = loop._drop_refuted(CARD_YAML, "merge")
        self.assertNotIn("慢 2.12×", once)
        twice = loop._drop_refuted(once, "merge")
        self.assertEqual(once, twice)

    def test_drop_refuted_keeps_other_entries(self):
        text = CARD_YAML + "  - id: kv4\n    why: \"慢 1.73×\"\n"
        out = loop._drop_refuted(text, "merge")
        self.assertIn("慢 1.73×", out)
        self.assertIn("kv4", out)

    def test_set_field_replaces_multiline_list(self):
        text = loop._set_field(CARD_YAML, loop._cand_block(CARD_YAML, "kv4"), 4,
                               "fails", ["2026-10-28 a", "2026-10-28 b"])
        seg = text[slice(*loop._cand_block(text, "kv4"))]
        self.assertIn("    fails:", seg)
        self.assertIn('      - "2026-10-28 a"', seg)
        self.assertIn('      - "2026-10-28 b"', seg)
        # 幂等：再设一次不会叠加
        again = loop._set_field(text, loop._cand_block(text, "kv4"), 4, "fails", ["2026-10-28 c"])
        self.assertNotIn("2026-10-28 a", again)
        self.assertIn("2026-10-28 c", again)

    def test_set_field_appends_when_absent(self):
        text = loop._set_field(CARD_YAML, loop._cand_block(CARD_YAML, "kv4"), 4, "status", "exhausted")
        seg = text[slice(*loop._cand_block(text, "kv4"))]
        self.assertIn("    status: exhausted", seg)
        self.assertTrue(seg.rstrip().endswith("status: exhausted"), seg)

    def test_scalar_quoting(self):
        self.assertEqual(loop._scalar("exhausted"), "exhausted")
        self.assertEqual(loop._scalar("impl"), "impl")
        self.assertIn("\"", loop._scalar("慢 2.1×：spill 30"))


class TestRecords(unittest.TestCase):
    def test_append_and_read_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            card = Path(td)
            self.assertEqual(loop.read_records(card), [])
            loop.append_record(card, _record(tag="a"))
            loop.append_record(card, _record(tag="b", decision="INCONCLUSIVE"))
            recs = loop.read_records(card)
        self.assertEqual([r["tag"] for r in recs], ["a", "b"])
        self.assertEqual(recs[1]["decision"], "INCONCLUSIVE")

    def test_broken_jsonl_line_is_ignored(self):
        with tempfile.TemporaryDirectory() as td:
            card = Path(td)
            d = loop.db_dir(card)
            (d / "20260101.jsonl").write_text('{"tag": "ok", "skv": 1}\nnot-json\n')
            recs = loop.read_records(card)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["tag"], "ok")

    def test_load_candidates_without_file_is_empty(self):
        with tempfile.TemporaryDirectory() as td:
            d = loop.load_candidates(Path(td))
        self.assertEqual(d["candidates"], [])
        self.assertEqual(d["refuted"], [])
        self.assertEqual(d["directions"], [])


class TestBudgetPure(unittest.TestCase):
    """方向预算的前置条件（纯字典，不需要 PyYAML / 文件）。"""

    def _cands(self, candidates, directions=None):
        return {"defaults": {}, "directions": directions or [], "candidates": candidates,
                "refuted": []}

    def test_empty_queue_has_no_problem(self):
        self.assertEqual(loop.budget_problems(self._cands([])), [])

    def test_candidate_without_direction_is_a_problem(self):
        probs = loop.budget_problems(self._cands([{"id": "a", "status": "pending"}]))
        self.assertEqual(len(probs), 1)
        self.assertIn("没声明 direction", probs[0])

    def test_unknown_direction_is_a_problem(self):
        probs = loop.budget_problems(self._cands(
            [{"id": "a", "status": "pending", "direction": "typo"}],
            [{"id": "real", "status": "open", "refutes": 0}]))
        self.assertIn("不在 directions 段里", probs[0])

    def test_exhausted_direction_is_a_problem(self):
        probs = loop.budget_problems(self._cands(
            [{"id": "a", "status": "pending", "direction": "d"}],
            [{"id": "d", "status": "open", "refutes": loop.MAX_DIRECTION_REFUTES}]))
        self.assertIn("已证伪", probs[0])

    def test_declared_exhausted_status_is_enough(self):
        probs = loop.budget_problems(self._cands(
            [{"id": "a", "status": "pending", "direction": "d"}],
            [{"id": "d", "status": "exhausted", "refutes": 0}]))
        self.assertEqual(len(probs), 1)

    def test_one_refute_still_open(self):
        self.assertEqual(loop.budget_problems(self._cands(
            [{"id": "a", "status": "pending", "direction": "d"}],
            [{"id": "d", "status": "open", "refutes": 1}])), [])

    def test_adopted_candidates_are_exempt(self):
        # adopted 的候选不再「生成」，不消耗也不受方向预算约束
        self.assertEqual(loop.budget_problems(self._cands(
            [{"id": "a", "status": "adopted"}])), [])

    def test_tag_filter(self):
        c = self._cands([{"id": "good", "status": "pending", "direction": "d"},
                         {"id": "bad", "status": "pending"}],
                        [{"id": "d", "status": "open", "refutes": 0}])
        self.assertEqual(loop.budget_problems(c, tags={"good"}), [])
        self.assertEqual(len(loop.budget_problems(c, tags={"bad"})), 1)

    def test_impl_fails_at_cap_is_a_problem(self):
        probs = loop.budget_problems(self._cands(
            [{"id": "a", "status": "pending", "direction": "d",
              "fails": ["x"] * loop.MAX_IMPL_FAILS}],
            [{"id": "d", "status": "open", "refutes": 0}]))
        self.assertIn("判死线", probs[0])


@unittest.skipUnless(
    HAVE_YAML,
    "未安装 PyYAML：本用例需先经 candidates.yaml 读出 refuted 清单，无依赖环境下无法构造；"
    "判定与文本改写用例不受影响",
)
class TestRunRefusesRefuted(unittest.TestCase):
    """队列推进的硬规则：已判死的 tag 不许重跑（这一步不需要 GPU，也不起子进程）。"""

    def test_refuted_tag_returns_2_without_running(self):
        with tempfile.TemporaryDirectory() as td:
            card = Path(td)
            (card / "candidates.yaml").write_text(CARD_YAML, encoding="utf-8")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = loop.cmd_run(SimpleNamespace(card=card, tags=["merge"], force=False))
            out = buf.getvalue()
        self.assertEqual(rc, 2, out)
        self.assertIn("拒绝重跑", out)
        self.assertIn("spill 30", out)


@unittest.skipUnless(
    HAVE_YAML,
    "未安装 PyYAML：loop.py 的 candidates.yaml 读写走 yaml.safe_load，无法在无依赖环境验证；"
    "纯判定与文本改写用例已由其它测试覆盖",
)
class TestCandidatesFile(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.card = Path(self._td.name)
        (self.card / "candidates.yaml").write_text(CARD_YAML, encoding="utf-8")
        # 第 3 步的产物：真实卡由 derive.py 生成 roofline.json。
        # 没有它 loop.py run 会拒跑（bottleneck_problems），
        # 所以测试卡也要有这个文件，否则每个 run 测试都在测「门禁拦住了」。
        import json
        (self.card / "roofline.json").write_text(
            json.dumps({"bound": "mixed",
                        "evidence": "测试用：derive.py bound=mixed"}), encoding="utf-8")

    def tearDown(self):
        self._td.cleanup()

    def _write(self, text):
        (self.card / "candidates.yaml").write_text(text, encoding="utf-8")

    def _cands(self):
        return loop.load_candidates(self.card)

    def _quiet(self, fn, *a, **kw):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = fn(*a, **kw)
        return rc, buf.getvalue()

    # ---------------------------------------------------------- impl-broken

    def test_impl_broken_counts_without_touching_direction(self):
        before = loop.direction_table(self._cands())["split-k"]["refutes"]
        rc, out = self._quiet(loop.cmd_impl_broken,
                              SimpleNamespace(card=self.card, tag="kv4", why="锚点没落上，算错"))
        self.assertEqual(rc, 0, out)
        d = self._cands()
        kv4 = {c["id"]: c for c in d["candidates"]}["kv4"]
        self.assertEqual(len(kv4["fails"]), 1)
        self.assertIn("锚点没落上", kv4["fails"][0])
        # * 核心断言：实现失败不吃方向预算
        self.assertEqual(loop.direction_table(d)["split-k"]["refutes"], before)
        self.assertNotIn("kv4", [r["id"] for r in d["refuted"]])
        self.assertIn("方向预算没动", out)

    def test_impl_broken_three_times_kills_only_the_impl(self):
        for i in range(loop.MAX_IMPL_FAILS):
            rc, out = self._quiet(loop.cmd_impl_broken,
                                  SimpleNamespace(card=self.card, tag="kv4", why="第 %d 次" % (i + 1)))
            self.assertEqual(rc, 0, out)
        d = self._cands()
        self.assertNotIn("kv4", [c["id"] for c in d["candidates"]])
        entry = {r["id"]: r for r in d["refuted"]}["kv4"]
        self.assertEqual(entry["kind"], "impl")
        self.assertIn("连挂 3 次", entry["why"])
        self.assertIn("第 3 次", entry["why"])
        # 方向依旧开着
        self.assertEqual(loop.direction_table(d)["split-k"]["refutes"], 0)
        self.assertEqual(loop.budget_problems(d), [])

    def test_impl_broken_undeclared_tag_refused(self):
        rc, out = self._quiet(loop.cmd_impl_broken,
                              SimpleNamespace(card=self.card, tag="nope", why="x"))
        self.assertEqual(rc, 2)
        self.assertIn("不在 candidates 段里", out)

    def test_impl_broken_is_logged(self):
        self._quiet(loop.cmd_impl_broken, SimpleNamespace(card=self.card, tag="kv4", why="算错"))
        recs = [r for r in loop.read_records(self.card) if r["decision"] == "IMPL_BROKEN"]
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["tag"], "kv4")
        self.assertEqual(recs[0]["direction"], "split-k")
        self.assertFalse(recs[0]["exhausted"])

    # ----------------------------------------------------- direction-refuted

    def test_direction_refuted_increments_and_kills_the_candidate(self):
        rc, out = self._quiet(loop.cmd_direction_refuted,
                              SimpleNamespace(card=self.card, tag="kv4", direction=None,
                                              why="机制：合并 buffer ⇒ 生命周期重叠 ⇒ 溢出"))
        self.assertEqual(rc, 0, out)
        d = self._cands()
        self.assertEqual(loop.direction_table(d)["split-k"]["refutes"], 1)
        self.assertNotIn("kv4", [c["id"] for c in d["candidates"]])
        entry = {r["id"]: r for r in d["refuted"]}["kv4"]
        self.assertEqual(entry["kind"], "direction")
        self.assertEqual(entry["direction"], "split-k")
        self.assertIn("回 op-constrain", out)

    def test_direction_refuted_twice_seals_the_direction(self):
        self._quiet(loop.cmd_direction_refuted,
                    SimpleNamespace(card=self.card, tag="kv4", direction=None, why="第一次"))
        self._quiet(loop.cmd_direction_refuted,
                    SimpleNamespace(card=self.card, tag="maxn32", direction="split-k",
                                    why="第二次（直接点名方向）"))
        d = self._cands()
        info = loop.direction_table(d)["split-k"]
        self.assertEqual(info["refutes"], loop.MAX_DIRECTION_REFUTES)
        self.assertEqual(info["status"], "exhausted")
        self.assertTrue(info["exhausted"])

    def test_sealed_direction_refuses_new_candidate(self):
        self._quiet(loop.cmd_direction_refuted,
                    SimpleNamespace(card=self.card, tag="kv4", direction=None, why="第一次"))
        self._quiet(loop.cmd_direction_refuted,
                    SimpleNamespace(card=self.card, tag=None, direction="split-k",
                                    why="第二次"))
        # 模型又在已封方向上生成了一个候选
        self._write((self.card / "candidates.yaml").read_text().replace(
            "\nrefuted:\n",
            "\n  - id: kv4_v2\n    direction: split-k\n    status: pending\n"
            "    hypothesis: \"再试一次\"\n\nrefuted:\n"))
        rc, out = self._quiet(loop.cmd_run, SimpleNamespace(card=self.card, tags=["kv4_v2"],
                                                            force=False))
        self.assertEqual(rc, 2, out)
        self.assertIn("已证伪", out)
        self.assertIn("op-constrain", out)
        rc_q, out_q = self._quiet(loop.cmd_queue, SimpleNamespace(card=self.card))
        self.assertEqual(rc_q, 2)
        self.assertIn("kv4_v2", out_q)

    def test_one_refute_still_allows_a_new_candidate(self):
        self._quiet(loop.cmd_direction_refuted,
                    SimpleNamespace(card=self.card, tag="kv4", direction=None, why="第一次"))
        self._write((self.card / "candidates.yaml").read_text().replace(
            "\nrefuted:\n",
            "\n  - id: kv4_v2\n    direction: split-k\n    status: pending\n"
            "    hypothesis: \"换一种合并方式\"\n\nrefuted:\n"))
        rc, out = self._quiet(loop.cmd_queue, SimpleNamespace(card=self.card))
        self.assertEqual(rc, 0, out)

    def test_direction_refuted_creates_missing_direction(self):
        rc, out = self._quiet(loop.cmd_direction_refuted,
                              SimpleNamespace(card=self.card, tag=None, direction="brand_new",
                                              why="结构上不可能"))
        self.assertEqual(rc, 0, out)
        d = self._cands()
        self.assertEqual(loop.direction_table(d)["brand_new"]["refutes"], 1)
        self.assertIn("已建条目", out)

    def test_direction_refuted_without_direction_refused(self):
        self._write(CARD_YAML.replace("    direction: split-k\n    priority: 2\n",
                                      "    priority: 2\n"))
        rc, out = self._quiet(loop.cmd_direction_refuted,
                              SimpleNamespace(card=self.card, tag="kv4", direction=None, why="x"))
        self.assertEqual(rc, 2)
        self.assertIn("没声明 direction", out)

    def test_direction_refuted_unknown_tag_needs_direction(self):
        rc, out = self._quiet(loop.cmd_direction_refuted,
                              SimpleNamespace(card=self.card, tag="ghost", direction=None, why="x"))
        self.assertEqual(rc, 2)
        self.assertIn("不在 candidates 段里", out)

    # ------------------------------------------------------------------ run

    def test_force_still_bypasses_the_refuted_gate(self):
        # --force 是显式否决：死马能被重跑，但记录里必须留痕（forced=true）
        rc, out = self._quiet(loop.cmd_run, SimpleNamespace(
            card=self.card, tags=["merge"], force=True, skv=[32768], rounds=1,
            s=None, topk=None, nvalid=None, heads=None, qscale=None, tol=None,
            primary_qscale=None, stress_mult=None, no_acc=True))
        self.assertNotIn("拒绝重跑", out)
        self.assertIn("--force 重跑已判死项", out)

    def test_run_refuses_undeclared_tag(self):
        rc, out = self._quiet(loop.cmd_run, SimpleNamespace(card=self.card, tags=["ghost"],
                                                            force=False))
        self.assertEqual(rc, 2)
        self.assertIn("不在 candidates.yaml 的队列里", out)

    def test_run_refuses_candidate_without_direction(self):
        self._write(CARD_YAML.replace("    direction: split-k\n    priority: 2\n",
                                      "    priority: 2\n"))
        rc, out = self._quiet(loop.cmd_run, SimpleNamespace(card=self.card, tags=["kv4"],
                                                            force=False))
        self.assertEqual(rc, 2)
        self.assertIn("没声明 direction", out)

    def test_legacy_refute_explains_the_split(self):
        rc, out = self._quiet(loop.cmd_refute, SimpleNamespace(card=self.card, tag="kv4", why="x"))
        self.assertEqual(rc, 2)
        self.assertIn("impl-broken", out)
        self.assertIn("direction-refuted", out)

    # -------------------------------------------------------------- 其它命令

    def test_adopt_without_done_when_is_refused(self):
        """adopt 是 verify 的旧名：不跑命令就 adopted 的路径已经关掉。"""
        rc, out = self._quiet(loop.cmd_adopt, SimpleNamespace(card=self.card, tag="maxn32"))
        self.assertEqual(rc, 2, out)
        self.assertIn("done_when", out)
        got = {c["id"]: c for c in self._cands()["candidates"]}
        self.assertEqual(got["maxn32"]["status"], "pending")

    def test_adopt_unknown_tag_is_refused(self):
        rc, out = self._quiet(loop.cmd_adopt, SimpleNamespace(card=self.card, tag="brand_new"))
        self.assertEqual(rc, 2, out)
        self.assertIn("不在 candidates 段里", out)
        self.assertNotIn("brand_new", [c["id"] for c in self._cands()["candidates"]])

    def test_load_contract_tol_reads_contract_yaml(self):
        (self.card / "contract.yaml").write_text(
            "op: demo\ntolerance:\n  vs_fp32_ref:\n    max_abs: 2.0e-3\n", encoding="utf-8")
        self.assertAlmostEqual(loop.load_contract_tol(self.card), 2.0e-3)
        self.assertAlmostEqual(loop.load_contract_tol(self.card / "nope", default=1e-3), 1e-3)

    def test_queue_lists_pending_and_refuted(self):
        loop.append_record(self.card, _record(tag="maxn32"))
        rc, out = self._quiet(loop.cmd_queue, SimpleNamespace(card=self.card))
        self.assertEqual(rc, 0, out)
        self.assertIn("maxn32", out)
        self.assertIn("已跑过", out)
        self.assertIn("已判死", out)
        self.assertIn("spill 30", out)
        self.assertIn("方向预算", out)

    def test_queue_shows_impl_fails(self):
        self._quiet(loop.cmd_impl_broken, SimpleNamespace(card=self.card, tag="kv4", why="算错"))
        rc, out = self._quiet(loop.cmd_queue, SimpleNamespace(card=self.card))
        self.assertEqual(rc, 0, out)
        self.assertIn("1/3 次实现失败", out)

    def test_status_reports_best_and_pending(self):
        loop.append_record(self.card, _record(tag="maxn32", cand_ms=0.69))
        loop.append_record(self.card, _record(tag="kv4", cand_ms=0.74, decision="INCONCLUSIVE"))
        rc, out = self._quiet(loop.cmd_status, SimpleNamespace(card=self.card))
        self.assertIn("已判死 1", out)
        self.assertIn("最强", out)
        self.assertIn("maxn32", out)
        self.assertIn("kv4", out)          # 仍是 pending，出现在「下一步」
        self.assertIn("方向", out)

    def test_status_hides_candidates_blocked_by_budget(self):
        self._quiet(loop.cmd_direction_refuted,
                    SimpleNamespace(card=self.card, tag="kv4", direction=None, why="一"))
        self._quiet(loop.cmd_direction_refuted,
                    SimpleNamespace(card=self.card, tag=None, direction="split-k", why="二"))
        self._write((self.card / "candidates.yaml").read_text().replace(
            "\nrefuted:\n",
            "\n  - id: kv4_v2\n    direction: split-k\n    status: pending\n"
            "    hypothesis: \"再试\"\n\nrefuted:\n"))
        rc, out = self._quiet(loop.cmd_status, SimpleNamespace(card=self.card))
        self.assertIn("caveat 候选 kv4_v2", out)          # 拦住的原因要说出来
        nxt = out.split("下一步：", 1)[1]
        self.assertIn("maxn32", nxt)
        self.assertNotIn("kv4_v2", nxt)              # 别把人指向死路

    def test_status_on_empty_card(self):
        with tempfile.TemporaryDirectory() as td:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                loop.cmd_status(SimpleNamespace(card=Path(td)))
        self.assertIn("待做 0", buf.getvalue())

    def test_report_renders_table(self):
        # 默认 report 只渲染 claimable；给一条三门全过的记录
        rec = _record()
        rec.update({
            "caliber": {"what": "op_call", "alloc": "false", "thermal": "hot",
                        "shape": "s128k32768", "device": "gfx938"},
            "significant": True, "acc_ok": True,
        })
        loop.append_record(self.card, rec)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            loop.cmd_report(SimpleNamespace(card=self.card))
        out = buf.getvalue()
        self.assertIn("| SKV | tag |", out)
        self.assertIn("| 32768 | maxn32 |", out)

    def test_report_without_records_exits_1(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = loop.cmd_report(SimpleNamespace(card=self.card))
        self.assertEqual(rc, 1)

    def test_argparse_dispatch_status(self):
        from unittest import mock
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), \
                mock.patch("sys.argv", ["loop.py", "-C", str(self.card), "status"]):
            rc = loop.main()
        self.assertEqual(rc, 0)
        self.assertIn("候选", buf.getvalue())

    def test_argparse_dispatch_impl_broken(self):
        from unittest import mock
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), \
                mock.patch("sys.argv", ["loop.py", "-C", str(self.card),
                                        "impl-broken", "kv4", "--why", "算错"]):
            rc = loop.main()
        self.assertEqual(rc, 0)
        self.assertIn("1/3", buf.getvalue())


@unittest.skipUnless(
    HAVE_YAML,
    "未安装 PyYAML：done_when 的读取走 candidates.yaml，无依赖环境无法构造",
)
class TestDoneWhen(unittest.TestCase):
    """完成判定必须由外部命令裁决：`verify` 返回 0 才允许 adopted。

    被测的是**hard rule**，不是提示词：模型只要改 YAML 就能自我批准，这里让
    「改成 adopted」这条路径只能经过一次真实子进程的退出码。
    """

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.card = Path(self._td.name)
        (self.card / "candidates.yaml").write_text(CARD_YAML, encoding="utf-8")
        import json as _json
        (self.card / "roofline.json").write_text(
            _json.dumps({"bound": "mixed", "evidence": "测试用"}), encoding="utf-8")
        # 主张门的前置：adopted 必须有够格的 run 记录。
        # 这里放一条「三门全过」的记录，让本类专注测 done_when 机制本身；
        # 主张门自己的行为由 TestClaimGate 覆盖。
        _good = _record()
        _good.update({
            "caliber": {"what": "op_call", "alloc": "false", "thermal": "hot",
                        "shape": "s128k32768", "device": "gfx938"},
            "significant": True, "acc_ok": True,
        })
        loop.append_record(self.card, _good)

    def tearDown(self):
        self._td.cleanup()

    # ---------------------------------------------------------------- 工具
    def _write(self, text):
        (self.card / "candidates.yaml").write_text(text, encoding="utf-8")

    def _with_done_when(self, cmd, tag="maxn32"):
        """给某个候选插一条 done_when（JSON 串 == YAML 双引号标量，转义一致）。"""
        import json
        text = (self.card / "candidates.yaml").read_text(encoding="utf-8")
        blk = loop._cand_block(text, tag)
        text = loop._set_field(text, blk, 4, "done_when", cmd)
        self._write(text)

    def _status(self, tag="maxn32"):
        return {c["id"]: c for c in loop.load_candidates(self.card)["candidates"]}[tag].get("status")

    def _quiet(self, fn, *a, **kw):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = fn(*a, **kw)
        return rc, buf.getvalue()

    # ------------------------------------------------------- 纯判定（不落盘）
    def test_adopted_without_done_when_is_a_problem(self):
        probs = loop.done_when_problems(
            {"candidates": [{"id": "a", "status": "adopted", "direction": "d"}]})
        self.assertEqual(len(probs), 1)
        self.assertIn("done_when", probs[0])
        self.assertIn("外部裁决点", probs[0])

    def test_adopted_with_done_when_is_fine(self):
        self.assertEqual(loop.done_when_problems(
            {"candidates": [{"id": "a", "status": "adopted", "done_when": "exit 0"}]}), [])

    def test_pending_without_done_when_is_fine(self):
        # 还没到完成判定那一步；什么都不要求（verify 会另拦）
        self.assertEqual(loop.done_when_problems(
            {"candidates": [{"id": "a", "status": "pending"}]}), [])

    def test_done_when_problems_respect_tag_filter(self):
        cands = {"candidates": [{"id": "a", "status": "adopted"},
                                {"id": "b", "status": "adopted", "done_when": "exit 0"}]}
        self.assertEqual(len(loop.done_when_problems(cands, tags={"b"})), 0)
        self.assertEqual(len(loop.done_when_problems(cands, tags={"a"})), 1)

    # --------------------------------------------------------------- verify
    def test_verify_success_adopts_and_records_rc(self):
        self._with_done_when("exit 0")
        rc, out = self._quiet(loop.cmd_verify, SimpleNamespace(card=self.card, tag="maxn32"))
        self.assertEqual(rc, 0, out)
        self.assertEqual(self._status(), "adopted")
        got = {c["id"]: c for c in loop.load_candidates(self.card)["candidates"]}["maxn32"]
        self.assertIn("adopted_on", got)
        self.assertIn("expect", got)            # 其它字段不被改写
        recs = [r for r in loop.read_records(self.card) if r["decision"] == "VERIFIED"]
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["done_when_rc"], 0)
        self.assertEqual(recs[0]["tag"], "maxn32")

    def test_verify_failure_keeps_status_and_prints_tail_only(self):
        self._with_done_when("for i in $(seq 1 30); do echo line$i; done; exit 3")
        rc, out = self._quiet(loop.cmd_verify, SimpleNamespace(card=self.card, tag="maxn32"))
        self.assertEqual(rc, 1, out)
        self.assertEqual(self._status(), "pending")     # 保持原状态
        self.assertIn("line30", out)                    # 尾部回显
        self.assertIn("line11", out)
        self.assertNotIn("line10\n", out)               # 只留最后 20 行
        self.assertIn("不许写 adopted", out)
        rec = [r for r in loop.read_records(self.card) if r["decision"] == "VERIFY_FAILED"][0]
        self.assertEqual(rec["done_when_rc"], 3)
        self.assertEqual(rec["status"], "pending")

    def test_verify_without_done_when_is_refused(self):
        rc, out = self._quiet(loop.cmd_verify, SimpleNamespace(card=self.card, tag="maxn32"))
        self.assertEqual(rc, 2, out)
        self.assertIn("没有 done_when", out)
        self.assertEqual(self._status(), "pending")

    def test_verify_unknown_tag_is_refused(self):
        rc, out = self._quiet(loop.cmd_verify, SimpleNamespace(card=self.card, tag="ghost"))
        self.assertEqual(rc, 2, out)
        self.assertIn("不在 candidates 段里", out)

    def test_verify_timeout_is_a_failure_not_a_pass(self):
        self._with_done_when("sleep 5")
        old = loop.VERIFY_TIMEOUT_S
        loop.VERIFY_TIMEOUT_S = 0.2
        try:
            rc, out = self._quiet(loop.cmd_verify, SimpleNamespace(card=self.card, tag="maxn32"))
        finally:
            loop.VERIFY_TIMEOUT_S = old
        self.assertEqual(rc, 1, out)
        self.assertEqual(self._status(), "pending")
        rec = [r for r in loop.read_records(self.card) if r["decision"] == "VERIFY_FAILED"][0]
        self.assertEqual(rec["done_when_rc"], 124)

    def test_verify_exposes_card_dir_to_the_command(self):
        # 命令在仓库根执行；卡目录通过 $OPS_CARD 暴露（不然卡片相对路径全都失效）
        self._with_done_when('test -f "$OPS_CARD/candidates.yaml"')
        rc, out = self._quiet(loop.cmd_verify, SimpleNamespace(card=self.card, tag="maxn32"))
        self.assertEqual(rc, 0, out)
        self.assertEqual(self._status(), "adopted")

    # ------------------------------------------------------- run 前置校验
    def test_run_refuses_adopted_candidate_without_done_when(self):
        text = (self.card / "candidates.yaml").read_text(encoding="utf-8")
        text = loop._set_field(text, loop._cand_block(text, "maxn32"), 4, "status", "adopted")
        self._write(text)
        rc, out = self._quiet(loop.cmd_run, SimpleNamespace(card=self.card, tags=["maxn32"],
                                                           force=False))
        self.assertEqual(rc, 2, out)
        self.assertIn("拒绝跑", out)
        self.assertIn("done_when", out)

    def test_run_does_not_refuse_a_pending_candidate_for_done_when(self):
        # pending 还没有完成判定；它在这里被拒是因为没有 bench 适配器，不是 done_when
        rc, out = self._quiet(loop.cmd_run, SimpleNamespace(card=self.card, tags=["maxn32"],
                                                           force=False))
        self.assertEqual(rc, 2, out)
        self.assertIn("A/B 适配器", out)
        self.assertNotIn("done_when", out)

    def test_verify_record_does_not_break_report(self):
        loop.append_record(self.card, _record())
        self._with_done_when("exit 0")
        self._quiet(loop.cmd_verify, SimpleNamespace(card=self.card, tag="maxn32"))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = loop.cmd_report(SimpleNamespace(card=self.card))
        self.assertEqual(rc, 0)
        self.assertIn("| 32768 | maxn32 |", buf.getvalue())

    def test_argparse_dispatch_verify(self):
        from unittest import mock
        self._with_done_when("exit 0")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), \
                mock.patch("sys.argv", ["loop.py", "-C", str(self.card),
                                        "verify", "--tag", "maxn32"]):
            rc = loop.main()
        self.assertEqual(rc, 0)
        self.assertIn("status=adopted", buf.getvalue())


@unittest.skipUnless(HAVE_YAML, "需要 PyYAML 读真实卡片")
class TestRealCard(unittest.TestCase):
    """参考实现 op-flash-mla 的 candidates.yaml 自己必须过门槛。

    这条测的是**产物**而不是脚本：卡片写歪了，`loop.py` 会在下次 run 时才拒绝，
    而那时人已经在卡上了。静态先拦一次。
    """

    CARD = REPO / "Plugins" / "ops" / "skills" / "op-flash-mla"

    def setUp(self):
        if not (self.CARD / "candidates.yaml").exists():
            self.skipTest("没有参考卡片")

    def test_card_has_no_budget_violation(self):
        d = loop.load_candidates(self.CARD)
        self.assertEqual(loop.budget_problems(d), [])

    def test_every_refuted_entry_has_mechanism(self):
        d = loop.load_candidates(self.CARD)
        self.assertTrue(d["refuted"], "已判死清单不该是空的")
        for r in d["refuted"]:
            self.assertIn(r.get("kind", "impl"), ("impl", "direction", "claim"), r.get("id"))
            why = (r.get("why") or "").strip()
            # 「试过，慢」这类结论挡不住下一个同类想法；机制至少要有说明长度
            self.assertGreater(len(why), 20, "%s 的 why 太短，像结论不像机制" % r.get("id"))

    def test_directions_have_statement(self):
        for d in loop.load_candidates(self.CARD)["directions"]:
            self.assertTrue((d.get("statement") or "").strip(), "%s 缺 statement" % d.get("id"))

    def test_adopted_candidates_have_done_when(self):
        """产物级检查：参考卡片自己必须守住「adopted ⇒ 有外部裁决命令」。

        这条测的是卡片而不是脚本 —— 卡片写歪了，`run`/`verify` 要到下一次被拒时才发现。
        """
        d = loop.load_candidates(self.CARD)
        self.assertEqual(loop.done_when_problems(d), [])
        for c in d["candidates"]:
            if str(c.get("status") or "") == "adopted":
                self.assertTrue(str(c.get("done_when") or "").strip(),
                                "%s 是 adopted 但没有 done_when" % c.get("id"))


if __name__ == "__main__":
    unittest.main()

class TestNoiseRecordsNotKept(unittest.TestCase):
    """噪声过大的记录不该留在权威流水里。

    背景（2026-11-24 实测）：同一份代码在四张卡上跑 event_isolated，
    lightop 基准从 197µs 漂到 220~236µs，MAD 4.9~7.2% 全部超 5% 熔断线。
    judge() 正确地判了 INCONCLUSIVE —— 但**记录本身还是写进了 runs/loop/**。

    这类记录的问题不是"结论错"，是**它会污染后续统计**：
    D4 的轮次、calibrate 的 realized_med、recall 的检索结果都会把它算进去。

    判据：真实卡片的 runs/loop/*.jsonl 里，
    不许有 `measure_mode=event_isolated` 且 `decision=INCONCLUSIVE`
    且判定理由是"噪声"的记录。要么重跑，要么不写。
    """

    def test_noise_is_quarantined_not_written_to_authoritative_log(self):
        """**反向**：写入端必须真的隔离噪声记录 —— 不能只靠上面那条扫描测试。

        上面的 test_no_noise_records_in_real_cards 是「事后扫描」；如果写入端
        不拦，它只能报错，不能防止污染。这一条直接调用 append_record，
        断言噪声记录进 noise/、非噪声记录进权威流。
        """
        import json
        import tempfile
        tmp = Path(tempfile.mkdtemp()) / "op-x"
        tmp.mkdir(parents=True)

        noisy = dict(_record(tag="noisy", decision="INCONCLUSIVE"),
                     why="baseline MAD/median=7.8% > 5%，本轮不可信（重跑而不是决策）")
        loop.append_record(tmp, noisy)
        loop.append_record(tmp, _record(tag="clean", decision="REJECTED_NO_GAIN"))

        auth = list((loop.db_dir(tmp)).glob("*.jsonl"))
        got = [json.loads(l) for f in auth for l in f.read_text().splitlines() if l.strip()]
        self.assertEqual([r["tag"] for r in got], ["clean"],
                         "噪声记录进了权威流水：%s" % [r["tag"] for r in got])
        self.assertTrue(any(r.get("quarantined") for r in got) is False)

        noise = list((loop.db_dir(tmp) / loop.NOISE_DIR).glob("*.jsonl"))
        self.assertEqual(len(noise), 1, "噪声记录没有被隔离留证")
        nq = [json.loads(l) for l in noise[0].read_text().splitlines() if l.strip()]
        self.assertEqual([r["tag"] for r in nq], ["noisy"])
        self.assertIn("quarantined", nq[0], "隔离后必须留痕，否则看起来像丢数据")

    def test_no_noise_records_in_real_cards(self):
        import json
        skills = Path(__file__).resolve().parents[1] / "skills"
        bad = []
        for f in skills.rglob("runs/loop/*.jsonl"):
            for line in f.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if (r.get("decision") == "INCONCLUSIVE"
                        and str(r.get("measure_mode")) == "event_isolated"
                        and "不可信" in str(r.get("why") or "")):
                    bad.append((str(f.relative_to(skills)), r.get("ts"), r.get("why")[:50]))
        self.assertEqual(bad, [],
                         "以下记录是噪声过大且不可 claim 的，不该留在权威流水里：%s" % bad)
