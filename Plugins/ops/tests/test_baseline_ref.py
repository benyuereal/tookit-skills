#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""图谱可靠性：必须能「指出基准在哪」，不是只知道「有库」。

业主原话：「我们现在建立的算子图谱你需要看看是否可靠，精细到算子级别，
这样 toolkit-skills 就知道从哪里找基准。」

审计发现（2026-11-23，102 个 impl）：
  - import_path 89 条全部可导入 ✅
  - 但 role 字段 **0/102** —— 分不清谁是厂商实现、谁是无优化参考
  - gemm-bf16 把 `torch.mm` 标 tier=primary，直接违反自己的 forbidden_impl_patterns
  - 同样这张卡的 fallback 是 `deepgemm.fp8_gemm`——fp8 不是 bf16，算子都不对

她要的是 `baseline_ref`：**bench 时拿哪条 impl 当性能基准**。
没有它，opbench 只能从图谱里读到 `torch.mm` 或 `triton.jit`，
那测出来的就是「我们的 kernel vs 未优化参考」——等于把 3.65× 那个假数再犯一次。

本测试守：
  A. 102/102 impl 都有 role
  B. tier=primary 的不能是 template / correctness_only（DSL 入口不能当主实现）
  C. 46/46 卡都有 baseline_ref，且 impl 指向 role ∈ {competitor, best_known}
  D. traffic_passes 必填（否则 201MB 与 268MB 混算）
  E. gemm-bf16 的 torch.mm 必须是 correctness_only 且不是 primary
"""
import glob
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CARDS = REPO / "Plugins/ops/op-atlas/cards"
VALIDATE = REPO / "Plugins/ops/op-atlas/validate.py"

VALID_ROLES = {"competitor", "best_known", "correctness_only", "template",
               "unverified", "wrong_op"}
BASELINE_ROLES = {"competitor", "best_known"}


def _cards():
    import yaml
    out = []
    for p in sorted(CARDS.glob("*.yaml")):
        out.append((p.stem, yaml.safe_load(p.read_text(encoding="utf-8")) or {}))
    return out


class TestBaselineRef(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import yaml  # noqa: F401
        except ImportError:                                    # pragma: no cover
            raise unittest.SkipTest("PyYAML 未安装")

    def test_every_impl_has_valid_role(self):
        bad, n = [], 0
        for cid, d in _cards():
            for i in d.get("impls") or []:
                n += 1
                r = i.get("role")
                if r not in VALID_ROLES:
                    bad.append("%s: %s role=%r" % (cid, i.get("import_path"), r))
        self.assertEqual(bad, [], "以下 impl 的 role 非法或缺失：\n" + "\n".join(bad[:10]))
        self.assertEqual(n, 105, "impl 总数应为 105（含 gemm-bf16 新增的 3 条）")

    def test_primary_is_never_template_or_correctness_only(self):
        bad = []
        for cid, d in _cards():
            for i in d.get("impls") or []:
                if i.get("tier") == "primary" and i.get("role") in ("template",
                                                                    "correctness_only",
                                                                    "wrong_op"):
                    bad.append("%s: %s (%s)" % (cid, i.get("import_path"), i.get("role")))
        self.assertEqual(bad, [], "DSL 入口/参考实现不能当 primary：\n" + "\n".join(bad))

    def test_every_card_has_baseline_ref(self):
        missing = [cid for cid, d in _cards() if not d.get("baseline_ref")]
        self.assertEqual(missing, [], "以下卡缺 baseline_ref：%s" % missing)

    def test_baseline_ref_points_to_allowed_role(self):
        bad = []
        for cid, d in _cards():
            br = d.get("baseline_ref") or {}
            if br.get("no_baseline_reason"):
                continue
            role = br.get("role")
            if role not in BASELINE_ROLES:
                bad.append("%s: baseline_ref.role=%r" % (cid, role))
        self.assertEqual(bad, [], "\n".join(bad))

    def test_baseline_ref_has_traffic_passes(self):
        bad = []
        for cid, d in _cards():
            br = d.get("baseline_ref") or {}
            if br.get("no_baseline_reason"):
                continue
            tp = br.get("traffic_passes")
            if not isinstance(tp, dict) or tp.get("passes") is None:
                bad.append(cid)
        self.assertEqual(bad, [],
                         "以下卡 baseline_ref 缺 traffic_passes（会把 201MB/268MB 混算）：%s" % bad)

    def test_gemm_bf16_torch_is_correctness_only(self):
        """gemm-bf16 曾把 torch.mm 标 primary —— 这直接违反了 baseline-policy。"""
        d = dict(_cards())["gemm-bf16"]
        torch_impls = [i for i in d["impls"] if i.get("repo") == "torch"]
        self.assertTrue(torch_impls, "gemm-bf16 应有 torch 条目作为正确性参考")
        for i in torch_impls:
            self.assertEqual(i.get("role"), "correctness_only")
            self.assertNotEqual(i.get("tier"), "primary",
                                "torch.mm 不能是 primary（未优化参考）")

    def test_gemm_bf16_has_real_competitor(self):
        """gemm-bf16 必须有真正的 bf16 competitor，且不能是 fp8 实现。"""
        d = dict(_cards())["gemm-bf16"]
        comp = [i for i in d["impls"] if i.get("role") == "competitor"]
        self.assertTrue(comp, "gemm-bf16 没有 competitor")
        for i in comp:
            ip = str(i.get("import_path") or "")
            self.assertNotIn("fp8", ip.lower(),
                             "fp8 实现不能当 bf16 GEMM 的基准：%s" % ip)

    def test_validate_clean(self):
        r = subprocess.run([sys.executable, "-I", str(VALIDATE),
                            "--atlas", str(CARDS.parent), "--repo", str(REPO)],
                           capture_output=True, text=True)
        self.assertNotRegex(r.stdout + r.stderr, r"\[E28\]")


if __name__ == "__main__":
    unittest.main()
