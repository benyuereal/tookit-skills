#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E24：ratio.value 必须声明量纲。

24 条实测曾经混着四种量纲：加速比 7 / 达成率 5 / 绝对吞吐 10 / 非收益比值 2。
`2.39`（倍）与 `887`（GB/s）放在同一字段，任何跨卡聚合都会错——
算不出全仓几何平均加速比，也排不出「哪张卡优化得最好」。

本测试守三件事：
  A. kind 缺失即 error（结构性拦截，不靠人记得写）
  B. kind=utilization 的 value 不许 >1.5（达成率超 150% 是分母口径错）
  C. aggregate.py 的几何平均**只用 speedup**——混入 throughput 会得出
     「平均 300 倍」这种荒谬数字
"""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
VALIDATE = REPO / "Plugins/ops/op-atlas/validate.py"
CARDS = REPO / "Plugins/ops/op-atlas/cards"


def _card_with(ratio_block):
    """造一张最小卡，把它塞进临时 atlas，返回 validate 的输出。"""
    import yaml
    d = tempfile.mkdtemp()
    atlas = Path(d) / "op-atlas"
    (atlas / "cards").mkdir(parents=True)
    card = {
        "id": "probe", "title": "probe", "family": "gemm", "math": "x",
        "status": "measured", "layer": "kernel", "op_card": None,
        "impls": [{"repo": "torch", "tier": "primary", "best_for": ["x"], "not_for": ["y"],
                   "pros": ["p"], "cons": ["c"], "source": "s",
                   "import_path": "torch.mm", "signature": "torch.mm(a,b)",
                   "smoke_test": 'python3 -I -c "import torch"'}],
        "ceiling": {"kind": "missing", "ref": None, "bound": "compute", "note": "n"},
        "shapes": ["s1"],
        "last_verdict": {"status": "measured", "kb_path": None,
                         "ratio": ratio_block, "conclusion": "测试用", "verified_on": "2026-11-23"},
    }
    (atlas / "cards" / "probe.yaml").write_text(
        yaml.dump(card, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (atlas / "INDEX.md").write_text("| x |\n|---|\n| [`probe`](cards/probe.yaml) |\n")
    r = subprocess.run([sys.executable, "-I", str(VALIDATE), "--atlas", str(atlas),
                        "--repo", str(REPO)], capture_output=True, text=True)
    return r.stdout + r.stderr


BASE_SCOPE = {"cards": 1, "in_len": "s1", "out_len": "s1", "warmup": "hot", "note": "n"}


class TestRatioKind(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import yaml  # noqa: F401
        except ImportError:                                    # pragma: no cover
            raise unittest.SkipTest("PyYAML 未安装")

    def test_missing_kind_is_error(self):
        out = _card_with({"value": 2.39, "vs": "相对基线的耗时比", "scope": dict(BASE_SCOPE)})
        self.assertIn("E24", out)
        self.assertIn("缺 kind", out)

    def test_utilization_over_150pct_is_error(self):
        out = _card_with({"kind": "utilization", "unit": "ratio", "value": 8.87,
                          "vs": "占峰值", "scope": dict(BASE_SCOPE)})
        self.assertIn("E24", out)
        self.assertIn("> 1.5", out)

    def test_throughput_without_unit_is_caught(self):
        out = _card_with({"kind": "throughput", "value": 887.0,
                          "vs": "带宽 GB/s", "scope": dict(BASE_SCOPE)})
        self.assertIn("E24", out)

    def test_valid_speedup_passes(self):
        out = _card_with({"kind": "speedup", "unit": "x", "value": 2.39,
                          "vs": "相对基线的耗时比", "scope": dict(BASE_SCOPE)})
        self.assertNotIn("E24", out, out[-400:])

    def test_all_real_cards_declare_kind(self):
        """全仓 24 张 measured/tuned 卡都必须有 kind —— 这条不许回退。"""
        import yaml
        missing = []
        for p in sorted(CARDS.glob("*.yaml")):
            d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            lv = d.get("last_verdict") or {}
            if lv.get("status") not in ("measured", "tuned"):
                continue
            r = lv.get("ratio") or {}
            if r.get("kind") not in ("speedup", "utilization", "throughput", "ratio_other"):
                missing.append(p.stem)
        self.assertEqual(missing, [], "以下卡缺合法 kind：%s" % missing)


class TestAggregateOnlyUsesSpeedup(unittest.TestCase):
    def test_geomean_excludes_throughput(self):
        sys.path.insert(0, str(REPO / "scripts"))
        import importlib.util
        spec = importlib.util.spec_from_file_location("_agg", str(REPO / "scripts/aggregate.py"))
        agg_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(agg_mod)
        r = subprocess.run([sys.executable, "-I", str(REPO / "scripts/aggregate.py"), "--selftest"],
                           capture_output=True, text=True)
        self.assertIn("selftest PASS", r.stdout, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
