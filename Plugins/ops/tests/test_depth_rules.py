#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E28：「到顶」必须能被推翻；「不推进」必须说明什么情况下会重启。

由来
----
2026-11-23 我在 rmsnorm 上试了 cross-lane-fold 后判死并标 at_floor，
后来发现 tilelang 能做到 0.82（我 0.78）——**那是 triton 的 codegen 上限，
不是方法极限**。

没有 falsifier 的「到顶」结论，等于把工具限制记成物理规律。
同一天全仓扫出 **11 张卡的 next_probe=surface:none** 且都没有 falsifier ——
这些卡实际上是「无声放弃」。

规则
----
  status=at_floor → 必须有 at_floor{kind ∈ {hardware_floor, impl_ceiling}, falsifier}
  next_probe.surface=none → 必须有 falsifier（或用 none_reason 写清「推翻/除非/若」）

本测试守：
  A. 去掉 falsifier → error
  B. at_floor 缺 kind → error
  C. at_floor 缺 falsifier → error
  D. 当前真实仓库 0 个 E28 error
  E. 全仓每张 surface=none 的卡都真的有 falsifier（不是靠 none_reason 里的关键字蒙混）
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
CARDS = ATLAS / "cards"


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


class TestDepthRules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import yaml  # noqa: F401
        except ImportError:                                    # pragma: no cover
            raise unittest.SkipTest("PyYAML 未安装")

    def test_current_repo_has_no_e28_errors(self):
        out = _validate()
        self.assertNotIn("[E28]", out, out[-900:])

    def test_surface_none_without_falsifier_is_error(self):
        import yaml
        dst = _clone()
        p = dst / "cards/gemm-bf16.yaml"
        d = yaml.safe_load(p.read_text(encoding="utf-8"))
        d["last_verdict"]["next_probe"].pop("falsifier", None)
        d["last_verdict"]["next_probe"]["none_reason"] = "已到地板"
        p.write_text(yaml.dump(d, allow_unicode=True, sort_keys=False), encoding="utf-8")
        out = _validate(dst)
        self.assertIn("[E28]", out)
        self.assertIn("无声放弃", out)

    def test_at_floor_requires_kind(self):
        import yaml
        dst = _clone()
        p = dst / "cards/rmsnorm-fused.yaml"
        d = yaml.safe_load(p.read_text(encoding="utf-8"))
        d["status"] = "at_floor"
        d["at_floor"] = {"basis": "x"}
        p.write_text(yaml.dump(d, allow_unicode=True, sort_keys=False), encoding="utf-8")
        out = _validate(dst)
        self.assertIn("[E28]", out)
        self.assertIn("at_floor.kind", out)

    def test_at_floor_requires_falsifier(self):
        import yaml
        dst = _clone()
        p = dst / "cards/rmsnorm-fused.yaml"
        d = yaml.safe_load(p.read_text(encoding="utf-8"))
        d["status"] = "at_floor"
        d["at_floor"] = {"kind": "impl_ceiling", "basis": "x"}
        p.write_text(yaml.dump(d, allow_unicode=True, sort_keys=False), encoding="utf-8")
        out = _validate(dst)
        self.assertIn("[E28]", out)
        self.assertIn("缺 falsifier", out)

    def test_every_surface_none_card_has_falsifier(self):
        """全仓扫描：surface=none 的卡必须有 falsifier 字段。"""
        import yaml
        bad = []
        for p in sorted(CARDS.glob("*.yaml")):
            d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            np_ = (d.get("last_verdict") or {}).get("next_probe") or {}
            if np_.get("surface") == "none" and not str(np_.get("falsifier") or "").strip():
                bad.append(p.stem)
        self.assertEqual(bad, [], "以下卡 surface=none 却没有 falsifier：%s" % bad)

    def test_at_floor_kind_is_one_of_two(self):
        import yaml
        p = CARDS / "rmsnorm-fused.yaml"
        d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        af = d.get("at_floor") or {}
        if af:
            self.assertIn(af.get("kind"), ("hardware_floor", "impl_ceiling"))


if __name__ == "__main__":
    unittest.main()
