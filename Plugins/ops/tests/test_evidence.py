#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E25：measured/tuned 卡的数字必须能追到可复跑的文件。

这是「分母漂移」的同族问题：数字写在卡片与 results 两处，只有一处对。
已经栽过两次（540 无来源、49.8 当峰值）。

分档（业主 2026-11-23 裁决）：
  hash_mismatch → **error**（真漂移，必须修，不许降级）
  missing       → **warning** + 不进聚合 + 登记到 gaps.md 的证据缺口节
                  连续 ≥2 批仍 missing → 降级 draft

为什么 missing 不立刻 error：14 张卡本来就没有可挂的证据文件，
强制 error 只会逼人回去造一个假 results 骗过门禁 —— 与「不为凑数造假」冲突。

本测试守四件事：
  A. 改了 results 但没更新卡片 → 哈希不符 → error
  B. 缺 evidence → warning（不是 error），且名单进 gaps.md
  C. aggregate.py 默认排除缺 evidence 的卡
  D. gaps.md 是生成物，与卡片一致（gen_gaps --check）
"""
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
VALIDATE = REPO / "Plugins/ops/op-atlas/validate.py"
CARDS = REPO / "Plugins/ops/op-atlas/cards"
GAPS = REPO / "Plugins/ops/op-atlas/gaps.md"
CARD = REPO / "Plugins/ops/op-atlas/cards/paged-attention-decode.yaml"
RESULTS = (REPO / "Plugins/ops/skills/op-optimize/cases"
           / "008-paged-decode-ceiling/results/paged-decode.json")


def _validate_atlas_dir(atlas):
    r = subprocess.run([sys.executable, "-I", str(VALIDATE), "--atlas", str(atlas),
                        "--repo", str(REPO)], capture_output=True, text=True)
    return r.stdout + r.stderr


def _validate():
    r = subprocess.run([sys.executable, "-I", str(VALIDATE),
                        "--atlas", str(REPO / "Plugins/ops/op-atlas"),
                        "--repo", str(REPO)], capture_output=True, text=True)
    return r.stdout + r.stderr


class TestEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import yaml  # noqa: F401
        except ImportError:                                    # pragma: no cover
            raise unittest.SkipTest("PyYAML 未安装")
        if not RESULTS.is_file():
            raise unittest.SkipTest("测试前提：paged-decode.json 应存在")

    def test_hash_mismatch_is_error(self):
        """卡片的 sha256 与实际文件不符 → 必须 error。

        在临时 atlas 上做：只改卡片里记录的 sha256，不动任何真文件。
        早期版本改仓内 results 再 try/finally 恢复，测试进程被 timeout 杀掉时
        （实测发生过两次）文件会永久留在 {} 状态，下一轮 validate 报「哈希不符」，
        而真因是测试没跑完。
        """
        import shutil
        import tempfile
        d = Path(tempfile.mkdtemp())
        try:
            atlas = d / "op-atlas"
            shutil.copytree(REPO / "Plugins/ops/op-atlas", atlas)
            card = atlas / "cards" / "paged-attention-decode.yaml"
            import yaml
            c = yaml.safe_load(card.read_text(encoding="utf-8"))
            ev = c["last_verdict"]["evidence"][0]
            ev["sha256"] = "0" * 64          # 与真实文件必然不符
            card.write_text(yaml.dump(c, allow_unicode=True, sort_keys=False),
                            encoding="utf-8")
            out = _validate_atlas_dir(atlas)
            self.assertIn("哈希不符", out)
            self.assertIn("paged-attention-decode", out)
            self.assertRegex(out, r"结果：[1-9]\d* 错误")
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_restore_clears_error(self):
        out = _validate()
        m = __import__("re").search(r"结果：(\d+) 错误", out)
        self.assertEqual(int(m.group(1)), 0, out[-800:])

    def test_missing_evidence_is_warning_not_error(self):
        """缺 evidence 只警告，不阻断（业主裁决）。"""
        out = _validate()
        self.assertIn("missing_evidence", out)
        # 不写死张数：随卡补证据会变（14 → 13 → …）。只要求「报了数且是 warning」
        self.assertRegex(out, r"\d+ 张卡缺 evidence")
        m = __import__("re").search(r"结果：(\d+) 错误", out)
        self.assertEqual(int(m.group(1)), 0,
                         "缺 evidence 不许报 error——那会逼人造假 results")

    def test_gaps_lists_every_missing_card(self):
        """裁决要求：缺证据必须机读列进 gaps.md，不许只打一行 stderr。"""
        txt = GAPS.read_text(encoding="utf-8")
        self.assertIn("证据缺口", txt)
        out = _validate()
        import re
        names = re.search(r"缺 evidence（warning，不进聚合）：(.+)", out)
        self.assertIsNotNone(names, out[-500:])
        listed = [x.strip() for x in names.group(1).split(",") if x.strip()]
        self.assertGreaterEqual(len(listed), 10, listed)
        for n in listed:
            self.assertIn("`%s`" % n, txt, "gaps.md 里没列 %s" % n)

    def test_aggregate_excludes_missing_evidence(self):
        sys.path.insert(0, str(REPO / "scripts"))
        import importlib.util
        spec = importlib.util.spec_from_file_location("_agg2", str(REPO / "scripts/aggregate.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        strict = m._load_cards()
        loose = m._load_cards(include_missing_evidence=True)
        self.assertLess(len(strict), len(loose),
                        "默认应排除缺 evidence 的卡（%d vs %d）" % (len(strict), len(loose)))
        for c in strict:
            self.assertTrue(c["id"], c)

    def test_gaps_is_generated_and_in_sync(self):
        r = subprocess.run([sys.executable, "-I", str(REPO / "scripts/gen_gaps.py"), "--check"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
