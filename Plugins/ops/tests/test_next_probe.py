#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E26：measured 必须说出「下一步改什么、改到什么时候停」。

问题：22 张 measured 里 0 张写了下一个攻击面 ⇒ 24 张卡退化成 24 张基线快照。

但只要求「字段存在」会变成形式主义——写「再看看」也能过机检。
所以三条硬要求（方案 §十.1 + 业主裁决）：

  surface     必须命中 surfaces.yaml 的 id，或用 `none` + none_reason
  hypothesis  必须含**一个数字**（预期可证伪）
  stop_if     必须含**机制词**（不涨时凭什么判死）

本测试守四件事：
  A. 缺 next_probe → error
  B. hypothesis 没数字 → error（这条防「再看看」）
  C. stop_if 没机制词 → error（这条防「不涨就停」）
  D. 全仓 24 张都有合法 next_probe
"""
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
VALIDATE = REPO / "Plugins/ops/op-atlas/validate.py"
CARDS = REPO / "Plugins/ops/op-atlas/cards"


def _probe_card(np_block):
    import yaml
    d = tempfile.mkdtemp()
    atlas = Path(d) / "op-atlas"
    (atlas / "cards").mkdir(parents=True)
    card = {
        "id": "probe", "title": "probe", "family": "gemm", "math": "x",
        "status": "measured", "layer": "kernel", "op_card": None,
        "variants": [{"repo": "torch", "tier": "primary", "best_for": ["x"], "not_for": ["y"],
                   "pros": ["p"], "cons": ["c"], "source": "s",
                   "import_path": "torch.mm", "signature": "torch.mm(a,b)",
                   "smoke_test": 'python3 -I -c "import torch"'}],
        "ceiling": {"kind": "missing", "ref": None, "bound": "compute", "note": "n"},
        "shapes": ["s1"],
        "last_verdict": {"status": "measured", "kb_path": None,
                         "ratio": {"kind": "speedup", "unit": "x", "value": 1.5,
                                   "vs": "相对基线",
                                   "scope": {"cards": 1, "in_len": "s1", "out_len": "s1",
                                             "warmup": "hot", "note": "n"}},
                         "conclusion": "测试", "verified_on": "2026-11-23"},
    }
    if np_block is not None:
        card["last_verdict"]["next_probe"] = np_block
    (atlas / "cards" / "probe.yaml").write_text(
        yaml.dump(card, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (atlas / "INDEX.md").write_text("| x |\n|---|\n| [`probe`](cards/probe.yaml) |\n")
    r = subprocess.run([sys.executable, "-I", str(VALIDATE), "--atlas", str(atlas),
                        "--repo", str(REPO)], capture_output=True, text=True)
    return r.stdout + r.stderr


GOOD = {"surface": "fusion-boundary",
        "hypothesis": "融合后应省一趟 HBM，提到 0.5 以上",
        "expected": "≥0.5",
        "stop_if": "若融合后不涨，说明瓶颈是跨 lane 归约 ⇒ 转 cross-lane-fold",
        "effort": "2h"}


class TestNextProbe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import yaml  # noqa: F401
        except ImportError:                                    # pragma: no cover
            raise unittest.SkipTest("PyYAML 未安装")

    def test_missing_next_probe_is_error(self):
        out = _probe_card(None)
        self.assertIn("E26", out)
        self.assertIn("缺 next_probe", out)

    def test_hypothesis_without_number_is_error(self):
        bad = dict(GOOD, hypothesis="再看看融合会不会更快")
        out = _probe_card(bad)
        self.assertIn("E26", out)
        self.assertIn("没有数字", out)

    def test_stop_if_without_mechanism_is_error(self):
        bad = dict(GOOD, stop_if="不涨就停")
        out = _probe_card(bad)
        self.assertIn("E26", out)
        self.assertIn("没有机制说明", out)

    def test_surface_must_be_known_id(self):
        bad = dict(GOOD, surface="不存在的攻击面")
        out = _probe_card(bad)
        self.assertIn("E26", out)
        self.assertIn("不在 surfaces.yaml", out)

    def test_surface_none_requires_reason(self):
        bad = dict(GOOD, surface="none")
        out = _probe_card(bad)
        self.assertIn("E26", out)
        self.assertIn("none_reason", out)

    def test_good_probe_passes(self):
        out = _probe_card(GOOD)
        self.assertNotIn("E26", out, out[-600:])

    def test_all_real_cards_have_valid_next_probe(self):
        """全仓 24 张 measured/tuned 卡都必须有合法 next_probe。"""
        import yaml
        bad = []
        for p in sorted(CARDS.glob("*.yaml")):
            d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            lv = d.get("last_verdict") or {}
            if lv.get("status") not in ("measured", "tuned"):
                continue
            np_ = lv.get("next_probe")
            if not isinstance(np_, dict):
                bad.append("%s: 缺 next_probe" % p.stem)
                continue
            if not np_.get("surface"):
                bad.append("%s: 缺 surface" % p.stem)
            if not re.search(r"\d", str(np_.get("hypothesis") or "")):
                bad.append("%s: hypothesis 无数字" % p.stem)
            # 机制词与 validate.py 的 STOP_MECHANISM_WORDS 保持一致
            if not any(w in str(np_.get("stop_if") or "")
                       for w in ("⇒", "因为", "由于", "机制", "说明", "证明", "→")):
                bad.append("%s: stop_if 无机制说明" % p.stem)
            if np_.get("surface") == "none" and not str(np_.get("none_reason") or "").strip():
                bad.append("%s: surface=none 但没写 none_reason" % p.stem)
        self.assertEqual(bad, [], "\n".join(bad))


if __name__ == "__main__":
    unittest.main()
