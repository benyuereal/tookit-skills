#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 3 步（定位瓶颈）的机检：没有 bottleneck 记录就不许跑候选。

2026-11-23 实测：`grep -nE "derive|bound|bottleneck" loop.py` **零结果**。
于是 6 步循环里，第 3 步是**唯一可以被整步跳过**的一步：

    agent 不跑 derive（不知道卡在算力还是内存）
         → 不跑 surfaces 的 check（没有证据）
         → 直接列候选 → 改代码 → 凭感觉说「快了」

这正是 op-optimize/SKILL.md 自己批评的「把闭环换成手搓」。

本测试守三件事：
  A. 缺 bottleneck 块 → 拒跑
  B. bottleneck 有 bound 但无 derive_evidence → 拒跑
  C. direction 的 class 与 bound 不匹配 → 拒跑
  D. bound 与 class 匹配时放行（防止拦得过头，把正常流程也堵死）
"""
import importlib.util
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LOOP = REPO / "Plugins/ops/engine/loop.py"


def _loop():
    spec = importlib.util.spec_from_file_location("_loop_bn", str(LOOP))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _cands(bottleneck=None, direction="layout-pack"):
    d = {"candidates": [{"id": "c1", "direction": direction, "status": "pending",
                         "hypothesis": "x"}]}
    if bottleneck is not None:
        d["bottleneck"] = bottleneck
    return d


class TestBottleneckGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loop = _loop()

    def test_missing_bottleneck_blocks_run(self):
        probs = self.loop.bottleneck_problems(_cands())
        self.assertTrue(probs, "缺 bottleneck 必须拒跑")
        self.assertIn("缺 bottleneck", probs[0])

    def test_bottleneck_without_evidence_blocks_run(self):
        probs = self.loop.bottleneck_problems(
            _cands({"bound": "memory", "derive_evidence": ""}))
        self.assertTrue(probs)
        self.assertTrue(any("derive_evidence" in p for p in probs), probs)

    def test_direction_class_mismatch_blocks_run(self):
        # bound=memory 只允许 residency/layout/memory-path；
        # precision-drop 属 precision class，必须被拒
        probs = self.loop.bottleneck_problems(
            _cands({"bound": "memory",
                    "derive_evidence": "derive.py: memory bound, mem 311us vs mma 145us"},
                   direction="precision-drop"))
        self.assertTrue(probs)
        self.assertTrue(any("不匹配" in p for p in probs), probs)

    def test_matching_direction_passes(self):
        probs = self.loop.bottleneck_problems(
            _cands({"bound": "memory",
                    "derive_evidence": "derive.py: memory bound"},
                   direction="layout-pack"))
        self.assertEqual(probs, [], "layout-pack 属 layout class，memory bound 允许它")

    def test_compute_bound_allows_precision(self):
        probs = self.loop.bottleneck_problems(
            _cands({"bound": "compute",
                    "derive_evidence": "derive.py: mma 145us < mem 311us"},
                   direction="precision-drop"))
        self.assertEqual(probs, [])

    def test_unknown_bound_is_rejected(self):
        probs = self.loop.bottleneck_problems(
            _cands({"bound": "不知道", "derive_evidence": "x"}))
        self.assertTrue(any("没有映射" in p for p in probs), probs)

    def test_unknown_direction_is_rejected(self):
        probs = self.loop.bottleneck_problems(
            _cands({"bound": "memory", "derive_evidence": "x"},
                   direction="我瞎编的方向"))
        self.assertTrue(any("不在 surfaces.yaml" in p for p in probs), probs)

    def test_bound_map_covers_all_six_classes(self):
        """bound-to-class.yaml 必须覆盖 surfaces.yaml 里的全部 6 个 class。"""
        import yaml
        b2c = yaml.safe_load((REPO / "Plugins/ops/skills/op-optimize/references"
                              / "bound-to-class.yaml").read_text(encoding="utf-8"))
        declared = set((b2c.get("class_members") or {}).keys())
        sur = yaml.safe_load((REPO / "Plugins/ops/skills/op-optimize/references"
                              / "surfaces.yaml").read_text(encoding="utf-8"))
        real = {s.get("class") for s in sur if isinstance(s, dict)}
        self.assertEqual(declared, real,
                         "class_members 与 surfaces.yaml 的 class 不一致：%s"
                         % (declared ^ real))
        # 每个 bound 的 classes 必须都是真实存在的 class
        for bound, node in (b2c.get("bound_to_class") or {}).items():
            extra = set(node.get("classes") or []) - real
            self.assertEqual(extra, set(), "bound=%s 引用了不存在的 class %s"
                             % (bound, extra))


if __name__ == "__main__":
    unittest.main()


class TestIsaPolicySurfaces(unittest.TestCase):
    """ISA 策略类攻击面：两条死路必须留在图里，且锚点可解析。

    2026-11-23 从 hbird-gemm 实测沉淀：
      slc-policy        buffer_load ... slc 慢 13%（默认 descriptor 最优）
      prefetch-depth-isa 源码级 prefetch 深度由 scheduler 决定，控制不了

    负结果比正结果值钱——留在 surfaces.yaml 是为了「查得到、别重试」。
    如果哪天有人把它们删了，下一轮就会再试一遍。
    """

    def test_dead_end_isa_surfaces_exist(self):
        import yaml
        p = REPO / "Plugins/ops/skills/op-optimize/references/surfaces.yaml"
        sur = yaml.safe_load(p.read_text(encoding="utf-8"))
        by_id = {s["id"]: s for s in sur if isinstance(s, dict)}
        for sid in ("slc-policy", "prefetch-depth-isa"):
            self.assertIn(sid, by_id, "%s 被删了——它是已验证的死路，删了会让人重试" % sid)
            self.assertEqual(by_id[sid].get("verdict"), "dead_end",
                             "%s 必须标 verdict: dead_end" % sid)
            self.assertIn("evidence", by_id[sid], "%s 缺 evidence（必须带实测数字）" % sid)

    def test_isa_surface_anchors_resolve(self):
        """law 的锚点必须能在那份文件里找到对应标题（E9 的提前版）。"""
        import re
        import yaml
        repo = REPO
        sur = yaml.safe_load((repo / "Plugins/ops/skills/op-optimize/references"
                              / "surfaces.yaml").read_text(encoding="utf-8"))
        bad = []
        for s in sur:
            if not isinstance(s, dict) or s.get("class") != "isa-policy":
                continue
            law = str(s.get("law") or "")
            fname, _, anchor = law.partition("#")
            target = repo / fname
            if not target.is_file():
                bad.append("%s: 文件不存在 %s" % (s["id"], fname))
                continue
            body = target.read_text(encoding="utf-8", errors="ignore")
            if not re.search(r"(?m)^#{1,4}\s+%s\b" % re.escape(anchor), body):
                bad.append("%s: 锚点 %r 找不到" % (s["id"], anchor))
        self.assertEqual(bad, [], "\n".join(bad))
