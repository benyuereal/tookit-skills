#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E27：性能基准必须是「也在做同一件优化过的活」。

由来
----
2026-11-23 我们用「triton 融合 vs torch 三段式」算出 3.65×，
这个数在工程上等于零——torch 三段式是**未优化的参考实现**，打赢它证明不了任何事。
24 张卡里 19 张「有数但没优化」，根子就在这里。

业主原话：「我们用 torch，这个基准肯定不行，这个不是一个优化的算子，
我们需要以 lightop 为基准。」

规则（`op-atlas/baseline-policy.yaml`）
--------------------------------------
  competitor / best_known  → 可以当性能基准
  correctness_only / reference / unverified → **不许**，且是 error 不是 warning

本测试守：
  A. 拿 ref-heavy 当基线 → error
  B. baseline 未声明 role → error
  C. 命中 forbidden_impl_patterns（torch-*）→ error
  D. 当前真实仓库 0 error（正向）
"""
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
VALIDATE = REPO / "Plugins/ops/op-atlas/validate.py"
ATLAS = REPO / "Plugins/ops/op-atlas"


def _validate(atlas=ATLAS):
    r = subprocess.run([sys.executable, "-I", str(VALIDATE),
                        "--atlas", str(atlas), "--repo", str(REPO)],
                       capture_output=True, text=True)
    return r.stdout + r.stderr


def _clone():
    d = tempfile.mkdtemp()
    dst = Path(d) / "op-atlas"
    shutil.copytree(ATLAS, dst)
    return dst


class TestBaselinePolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import yaml  # noqa: F401
        except ImportError:                                    # pragma: no cover
            raise unittest.SkipTest("PyYAML 未安装")

    def test_current_repo_has_no_baseline_errors(self):
        out = _validate()
        self.assertNotIn("[E27]", out, out[-900:])

    def test_ref_heavy_as_baseline_is_error(self):
        """拿 ref-heavy 当基线 → error（这是今天最典型的错误）。"""
        import yaml
        dst = _clone()
        p = dst / "cards/rmsnorm-fused.yaml"
        d = yaml.safe_load(p.read_text(encoding="utf-8"))
        d["last_verdict"]["comparison"].update({
            "baseline": "ref-heavy", "baseline_role": "correctness_only",
            "verdict": "相对 torch 参考快 3.65x"})
        p.write_text(yaml.dump(d, allow_unicode=True, sort_keys=False), encoding="utf-8")
        out = _validate(dst)
        self.assertIn("[E27]", out)
        self.assertIn("不能当性能基准", out)
        self.assertIn("命中禁止模式", out)

    def test_missing_baseline_role_is_error(self):
        import yaml
        dst = _clone()
        p = dst / "cards/gemm-fp8.yaml"
        d = yaml.safe_load(p.read_text(encoding="utf-8"))
        d["last_verdict"]["comparison"] = {"baseline": "aiter", "cand_us": 48,
                                           "base_us": 50, "ratio": 1.04,
                                           "inputs": 2, "outputs": 2,
                                           "alignment": {"note": "x"}}
        p.write_text(yaml.dump(d, allow_unicode=True, sort_keys=False), encoding="utf-8")
        out = _validate(dst)
        self.assertIn("未声明 baseline_role", out)

    def test_forbidden_pattern_matches_torch(self):
        """forbidden_impl_patterns 里的 torch-* 必须能匹配到。"""
        import yaml
        pol = yaml.safe_load((ATLAS / "baseline-policy.yaml").read_text(encoding="utf-8"))
        pats = pol.get("forbidden_impl_patterns") or []
        self.assertIn("ref-heavy", pats)
        self.assertIn("torch-*", pats)
        import fnmatch
        self.assertTrue(fnmatch.fnmatch("torch-native", "torch-*"))
        self.assertTrue(fnmatch.fnmatch("ref-heavy", "ref-heavy"))
        self.assertFalse(fnmatch.fnmatch("lightop-fused", "torch-*"))

    def test_policy_roles_are_explicit(self):
        import yaml
        pol = yaml.safe_load((ATLAS / "baseline-policy.yaml").read_text(encoding="utf-8"))
        roles = pol.get("roles") or {}
        for allowed in ("competitor", "best_known"):
            self.assertTrue(roles[allowed]["allowed_as_performance_baseline"], allowed)
        for denied in ("correctness_only", "reference", "unverified"):
            self.assertFalse(roles[denied]["allowed_as_performance_baseline"], denied)

    def test_variants_declare_role(self):
        """rmsnorm 的 5 个变体都要声明 role。"""
        import yaml
        d = yaml.safe_load((REPO / "Plugins/ops/skills/op-rmsnorm-fused/variants.yaml")
                           .read_text(encoding="utf-8"))
        for v in d["variants"]:
            self.assertIn("role", v, "变体 %s 缺 role" % v["name"])
        ref = next(v for v in d["variants"] if v["name"] == "ref-heavy")
        self.assertEqual(ref["role"], "correctness_only")


if __name__ == "__main__":
    unittest.main()
