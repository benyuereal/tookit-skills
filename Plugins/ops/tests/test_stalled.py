#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""stalled（变慢但结构变了）与 refuted（机制上不成立）必须分开。

来源：CompoundingKernel（arXiv:2611.19230）Quant/030 的最优路径 270 步里
有 76 步持平或更慢，最终仍从 6.25× 涨到 27.36×。
**最好的结果不是从一直变快的路径上长出来的。**

我们原来的机制相反：refuted 是唯一落点，MAX_DIRECTION_REFUTES=2 就封方向。
这会封掉踏脚石。现在拆成两类：

  refuted  机制上不成立（双流叠两 GEMM 同引擎，物理不可能）⇒ 计方向预算，可封
  stalled  变慢但结构变了 ⇒ **不计方向预算**，可当父代

本测试守四件事：
  A. stalled 不增加方向计数，方向不会被封
  B. stalled 必须带 what_changed（否则退化成「试过，慢」）
  C. stalled 条目的落点与 refuted 互斥（同 id 只能在一处）
  D. stalled 条目必须真被当过 parent —— 否则这套改动等于把 refuted 改名
"""
import importlib.util
import io
import pathlib
import tempfile
import unittest
from contextlib import redirect_stdout

REPO = pathlib.Path(__file__).resolve().parents[3]
LOOP = REPO / "Plugins/ops/skills/op-optimize/loop.py"


def _load_loop():
    spec = importlib.util.spec_from_file_location("_loop_under_test", str(LOOP))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BASE_YAML = """# 测试卡
defaults: {}
directions:
  - id: fuse-a
    statement: "把两段融进一个 kernel"
    status: open
    refutes: 1
candidates:
  - id: cand-b
    direction: fuse-a
    status: pending
    hypothesis: "换成 LDS 双缓冲"
"""


class TestStalledVsRefuted(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loop = _load_loop()
        try:
            import yaml  # noqa: F401
        except ImportError:                                    # pragma: no cover
            raise unittest.SkipTest("PyYAML 未安装")

    def _card(self, text=BASE_YAML):
        d = tempfile.mkdtemp()
        card = pathlib.Path(d) / "card"
        card.mkdir()
        (card / "candidates.yaml").write_text(text, encoding="utf-8")
        return card

    # ---- A：stalled 不计方向预算
    def test_stalled_does_not_consume_direction_budget(self):
        card = self._card()
        a = type("A", (), {"card": card, "tag": "cand-b", "why": "LDS 双缓冲更慢",
                           "what_changed": "buffer 从 1 份变 2 份", "ratio": "1.18",
                           "direction": None})()
        with redirect_stdout(io.StringIO()):
            rc = self.loop.cmd_stalled(a)
        self.assertEqual(rc, 0)

        import yaml
        d = yaml.safe_load((card / "candidates.yaml").read_text(encoding="utf-8"))
        info = {x["id"]: x for x in d["directions"]}["fuse-a"]
        self.assertEqual(info["refutes"], 1,
                         "stalled 不该增加方向计数（保持进 stalled 前的 1）")
        self.assertEqual(info["status"], "open", "stalled 不该封方向")
        self.assertEqual(len(d["stalled"]), 1)
        self.assertEqual(d["stalled"][0]["what_changed"], "buffer 从 1 份变 2 份")

    # ---- 对照：refuted 会计数并可能封方向
    def test_refuted_still_consumes_budget_and_can_exhaust(self):
        card = self._card()
        a = type("A", (), {"card": card, "tag": "cand-b", "direction": None,
                           "why": "同引擎双流叠两 GEMM，物理上不可能"})()
        with redirect_stdout(io.StringIO()):
            self.loop.cmd_direction_refuted(a)
        import yaml
        d = yaml.safe_load((card / "candidates.yaml").read_text(encoding="utf-8"))
        info = {x["id"]: x for x in d["directions"]}["fuse-a"]
        self.assertEqual(info["refutes"], 2, "refuted 必须计方向预算")
        self.assertEqual(info["status"], "exhausted",
                         "到 MAX_DIRECTION_REFUTES 就该封方向（refuted 的语义没变）")

    # ---- B：what_changed 必填
    def test_stalled_requires_what_changed(self):
        card = self._card()
        a = type("A", (), {"card": card, "tag": "cand-b", "why": "更慢",
                           "what_changed": "", "ratio": None, "direction": None})()
        with redirect_stdout(io.StringIO()):
            rc = self.loop.cmd_stalled(a)
        self.assertEqual(rc, 2, "没有 what_changed 必须拒绝——否则等于「试过，慢」")
        import yaml
        d = yaml.safe_load((card / "candidates.yaml").read_text(encoding="utf-8"))
        self.assertEqual(d.get("stalled") or [], [], "被拒时不该写入")

    # ---- C：落点互斥
    def test_stalled_and_refuted_are_mutually_exclusive(self):
        card = self._card()
        self.loop._write_candidates(
            card / "candidates.yaml",
            self.loop._to_stalled(BASE_YAML, "cand-b", "更慢", "结构变了", "fuse-a"))
        import yaml
        d = yaml.safe_load((card / "candidates.yaml").read_text(encoding="utf-8"))
        self.assertEqual([r["id"] for r in d["stalled"]], ["cand-b"])
        self.assertEqual(d.get("refuted") or [], [])

        # 再改成 refuted，stalled 里必须被清掉
        txt = (card / "candidates.yaml").read_text(encoding="utf-8")
        txt = self.loop._to_refuted(txt, "cand-b", "机制不成立", "direction", "fuse-a")
        self.loop._write_candidates(card / "candidates.yaml", txt)
        d = yaml.safe_load((card / "candidates.yaml").read_text(encoding="utf-8"))
        self.assertEqual([r["id"] for r in d["refuted"]], ["cand-b"])
        self.assertEqual(d.get("stalled") or [], [],
                         "转入 refuted 后 stalled 里不许留残影（否则同一 id 两处各说各话）")

    # ---- D：必须真被当过 parent
    def test_stalled_must_be_used_as_parent(self):
        import yaml
        d = yaml.safe_load(BASE_YAML)
        d["stalled"] = [{"id": "cand-b", "what_changed": "buffer 2 份", "why": "更慢"}]

        # 没有任何候选把它当 parent ⇒ 必须报出来
        probs = self.loop.stalled_parent_problems(copy_of(d))
        self.assertEqual(len(probs), 1)
        self.assertIn("从未被当过 parent", probs[0])

        # 有一个后续候选以它为 parent ⇒ 通过
        d2 = copy_of(d)
        d2["candidates"].append({"id": "cand-c", "parent": "cand-b",
                                 "direction": "fuse-a", "status": "pending"})
        self.assertEqual(self.loop.stalled_parent_problems(d2), [])

    def test_no_stalled_means_no_complaint(self):
        self.assertEqual(self.loop.stalled_parent_problems({"candidates": []}), [])


def copy_of(d):
    import copy
    return copy.deepcopy(d)


if __name__ == "__main__":
    unittest.main()


class TestCandidatesYamlPreservesComments(unittest.TestCase):
    """改 candidates.yaml 不许把注释吃掉。

    candidates.yaml 的注释不是装饰——它承载了「纪律」与 40+ 条判死的机制叙述，
    是这张卡最有价值的资产。用 yaml.safe_load→yaml.dump 往返会把它全丢掉
    （2026-11-23 实际发生：264 行带注释 → 343 行无注释，恢复靠 git checkout）。

    所以：loop.py 的所有写路径必须是**文本级**增删（_to_stalled/_to_refuted 用的
    就是这种方式），不许 round-trip。本测试守住这条。
    """

    def test_loop_does_not_roundtrip_yaml(self):
        import re
        src = LOOP.read_text(encoding="utf-8")
        # 找所有 yaml.dump 调用；它们只允许出现在测试/输出路径，不许出现在写卡片处
        dumps = [m.start() for m in re.finditer(r"yaml\.dump\(", src)]
        self.assertEqual(
            dumps, [],
            "loop.py 里出现 yaml.dump —— 写 candidates.yaml 时必须文本级增删，"
            "round-trip 会把全部注释吃掉")

    def test_writers_preserve_existing_comments(self):
        """真正的行为断言：写一次后，原有注释一行不少。"""
        loop = _load_loop()
        src = (
            "# 卡片级纪律：这三行注释必须活下来\n"
            "# 第二行\n"
            "defaults: {}\n"
            "directions:\n"
            "  - id: d1\n"
            "    statement: 'x'\n"
            "    status: open\n"
            "    refutes: 0\n"
            "candidates:\n"
            "  - id: c1\n"
            "    direction: d1\n"
            "    status: pending\n"
            "# refuted 段的说明\n"
            "refuted: []\n"
        )
        out = loop._to_stalled(src, "c1", "更慢", "结构变了", "d1")
        for line in ("# 卡片级纪律：这三行注释必须活下来", "# 第二行", "# refuted 段的说明"):
            self.assertIn(line, out, "写 stalled 时丢了注释：%s" % line)

        out2 = loop._to_refuted(out, "c1", "机制不成立", "direction", "d1")
        for line in ("# 卡片级纪律：这三行注释必须活下来", "# refuted 段的说明"):
            self.assertIn(line, out2, "写 refuted 时丢了注释：%s" % line)
