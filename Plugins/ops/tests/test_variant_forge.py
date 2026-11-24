#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""variant_forge 的测试：参数必须**真的接进去**，不能只是看起来改了。

被测核心不是「能生成文件」，是**生成物与 parent 在被改的锚点上不同、
且这个不同能被机器读回来**。反向用例和正向一样重要：未知参数 / 越界 /
锚点缺失必须报错，静默跳过会产出污染整轮结论的假变体。

测试不依赖 GPU / torch / triton：只做源码级变换与 AST 复核。
"""
import ast
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts/variant_forge.py"


def _load():
    spec = importlib.util.spec_from_file_location("_variant_forge", str(SCRIPT))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _card(dirpath, candidates, name="op-fake"):
    """在临时目录造一张最小 candidates.yaml。"""
    import yaml
    d = Path(dirpath) / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "candidates.yaml").write_text(
        yaml.safe_dump({"candidates": candidates}, allow_unicode=True,
                       sort_keys=False), encoding="utf-8")
    return d


class TestVariantForge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load()
        cls.tpl = cls.m.load_template("reduce-lastdim")
        cls.parent = cls.tpl.source

    # ------------------------------------------------------------ 正向

    def test_num_warps_patch_lands_in_source(self):
        r = self.m.forge(self.parent, {"num_warps": 8}, self.tpl, "parent")
        self.assertIn("num_warps=8", r.source)
        self.assertEqual(r.effective["num_warps"], 8,
                         "锚点上读回的值不是 8 —— 参数没接进去")
        self.assertTrue(r.changed)

    def test_block_patch_lands_in_launch_positional(self):
        r = self.m.forge(self.parent, {"BLOCK": 2048}, self.tpl, "parent")
        self.assertIn("2048", r.source)
        self.assertEqual(r.effective["BLOCK"], 2048)
        # launch 行里最后一个位置实参必须是 2048（不是 run() 里的 H）
        launch = [l for l in r.source.splitlines() if "rmsnorm_k[(" in l
                  and "x.stride(0)" in l][0]
        positional = launch.split("](", 1)[1].split(", num_warps=")[0]
        self.assertEqual(positional.rsplit(",", 1)[-1].strip(), "2048", launch)

    def test_two_patches_produce_different_files(self):
        a = self.m.forge(self.parent, {"num_warps": 4}, self.tpl, "parent")
        b = self.m.forge(self.parent, {"num_warps": 8}, self.tpl, "parent")
        self.assertNotEqual(a.source, b.source,
                            "num_warps=4 与 num_warps=8 生成了相同内容")
        self.assertIn("num_warps=4", a.source)
        self.assertIn("num_warps=8", b.source)

    def test_combined_patch_changes_two_lines(self):
        r = self.m.forge(self.parent, {"BLOCK": 2048, "num_warps": 8},
                         self.tpl, "parent")
        changed = [e for e in r.edits if e.old != e.new]
        self.assertEqual(len(changed), 2)
        self.assertEqual(r.effective, {"BLOCK": 2048, "num_warps": 8})

    def test_generated_source_parses_with_external_python(self):
        """用独立解释器做 ast.parse，证明生成物不是本进程的假象。"""
        r = self.m.forge(self.parent, {"BLOCK": 1024, "num_warps": 2},
                         self.tpl, "parent")
        d = Path(tempfile.mkdtemp())
        f = d / "v.py"
        f.write_text(r.source, encoding="utf-8")
        p = subprocess.run(
            [sys.executable, "-I", "-c",
             "import ast,sys; ast.parse(open(sys.argv[1],encoding='utf-8').read())",
             str(f)], capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        ast.parse(r.source)

    def test_diff_contains_changed_lines(self):
        r = self.m.forge(self.parent, {"num_warps": 8}, self.tpl, "parent")
        self.assertIn("-def run(x, w, eps=1e-6, num_warps=4):", r.diff)
        self.assertIn("+def run(x, w, eps=1e-6, num_warps=8):", r.diff)
        report = self.m.render_report(r)
        self.assertIn("num_warps", report)
        self.assertIn("4 → 8", report)

    def test_dry_run_writes_nothing(self):
        out = Path(tempfile.mkdtemp()) / "v.py"
        rc = self.m.main(["--parent", "reduce-lastdim", "--patch",
                          "num_warps=8", "--out", str(out), "--dry-run"])
        self.assertEqual(rc, 0)
        self.assertFalse(out.exists(), "--dry-run 写了文件")

    def test_cli_writes_file_when_not_dry_run(self):
        out = Path(tempfile.mkdtemp()) / "v.py"
        rc = self.m.main(["--parent", "reduce-lastdim", "--patch",
                          "num_warps=8", "--out", str(out)])
        self.assertEqual(rc, 0)
        self.assertTrue(out.exists())
        self.assertIn("num_warps=8", out.read_text(encoding="utf-8"))

    def test_alias_warps(self):
        r = self.m.forge(self.parent, {"warps": 8}, self.tpl, "parent")
        self.assertEqual(r.patches, {"num_warps": 8})
        self.assertIn("num_warps=8", r.source)

    # ------------------------------------------------------------ 反向

    def test_unknown_param_raises_and_lists_supported(self):
        with self.assertRaises(self.m.ForgeError) as cm:
            self.m.forge(self.parent, {"BM": 128}, self.tpl, "parent")
        msg = str(cm.exception)
        self.assertIn("不认识", msg)
        self.assertIn("BLOCK", msg)
        self.assertIn("num_warps", msg)

    def test_num_warps_zero_rejected(self):
        with self.assertRaises(self.m.ForgeError) as cm:
            self.m.forge(self.parent, {"num_warps": 0}, self.tpl, "parent")
        self.assertIn("越界", str(cm.exception))

    def test_num_warps_non_power_of_two_rejected(self):
        with self.assertRaises(self.m.ForgeError):
            self.m.forge(self.parent, {"num_warps": 3}, self.tpl, "parent")

    def test_missing_anchor_raises_not_silent(self):
        """构造一个改坏的模板：run() 没有 num_warps 参数。"""
        broken = self.parent.replace(
            "def run(x, w, eps=1e-6, num_warps=4):",
            "def run(x, w, eps=1e-6):")
        self.assertNotEqual(broken, self.parent, "构造坏模板失败")
        with self.assertRaises(self.m.ForgeError) as cm:
            self.m.forge(broken, {"num_warps": 8}, self.tpl, "broken")
        self.assertIn("锚点", str(cm.exception))

    def test_missing_launch_anchor_raises(self):
        """launch 调用被删掉 → BLOCK 找不到锚点，必须报错。"""
        broken = self.parent.replace(
            "    rmsnorm_k[(M,)](x, w, y, x.stride(0), H, eps, H, "
            "num_warps=num_warps)\n    return y",
            "    return y")
        self.assertNotEqual(broken, self.parent)
        with self.assertRaises(self.m.ForgeError) as cm:
            self.m.forge(broken, {"BLOCK": 2048}, self.tpl, "broken")
        self.assertIn("锚点", str(cm.exception))

    def test_verify_patch_rejects_unmodified_parent(self):
        """校验器本身要被校验：未改动的 parent 不能通过 verify_patch。"""
        with self.assertRaises(self.m.ForgeError):
            self.m.verify_patch(self.parent, {"num_warps": 8}, self.tpl)

    def test_non_integer_value_rejected(self):
        with self.assertRaises(self.m.ForgeError):
            self.m.forge(self.parent, {"BLOCK": "2048"}, self.tpl, "parent")

    def test_bad_patch_string_rejected(self):
        with self.assertRaises(self.m.ForgeError):
            self.m.parse_patch_arg("BLOCK")
        with self.assertRaises(self.m.ForgeError):
            self.m.parse_patch_arg("BLOCK=abc")

    # ------------------------------------------------------------ 卡片路径

    def test_card_candidate_forges(self):
        d = Path(tempfile.mkdtemp())
        card = _card(d, [{
            "id": "bm64",
            "forge": {"parent": "reduce-lastdim",
                      "patch": {"BLOCK": 2048, "num_warps": 8}},
        }])
        cand, spec = self.m.resolve_candidate(str(card), "bm64")
        self.assertEqual(cand["id"], "bm64")
        res = self.m.forge_candidate(str(card), "bm64", dry_run=True)
        self.assertEqual(res.effective["BLOCK"], 2048)
        self.assertEqual(res.effective["num_warps"], 8)

    def test_card_manual_rejected(self):
        """HIP 侧的候选标 forge: manual —— 必须明确拒绝，不生成假变体。"""
        d = Path(tempfile.mkdtemp())
        card = _card(d, [{"id": "hip-tile", "forge": "manual"}])
        with self.assertRaises(self.m.ForgeError) as cm:
            self.m.resolve_candidate(str(card), "hip-tile")
        self.assertIn("manual", str(cm.exception))

    def test_card_candidate_without_forge_raises(self):
        d = Path(tempfile.mkdtemp())
        card = _card(d, [{"id": "no-forge"}])
        with self.assertRaises(self.m.ForgeError) as cm:
            self.m.resolve_candidate(str(card), "no-forge")
        self.assertIn("forge", str(cm.exception))

    def test_real_card_reports_missing_candidate(self):
        """真实卡片上不存在的候选必须报错并列出已有候选。"""
        with self.assertRaises(self.m.ForgeError) as cm:
            self.m.resolve_candidate("op-rmsnorm-fused", "bm64")
        self.assertIn("fuse-residual", str(cm.exception))

    def test_selftest_passes(self):
        self.assertEqual(self.m.selftest(), 0)

    def test_failed_verification_no_file_written(self):
        """生成失败时 CLI 退 2，且不留下输出文件。"""
        out = Path(tempfile.mkdtemp()) / "v.py"
        rc = self.m.main(["--parent", "reduce-lastdim", "--patch",
                          "num_warps=0", "--out", str(out)])
        self.assertEqual(rc, 2)
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
