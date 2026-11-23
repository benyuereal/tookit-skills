#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T2 的反向测试：results json 不带齐 placement 就必须被拒。

为什么单独一个文件：本仓的历史教训是「判据写进去了却从未生效」
（同名函数定义两次、条件写成永远为假）。放置字段尤其如此——
一个只检查 `"placement" in rec` 的校验器，在 `vgpr: 0` 时照样放行，
而 vgpr=0 会让占用率公式静默变成「无限 CTA」，结论直接反向。

所以每个用例都构造**错的**输入，确认非 0 / 报错；再构造**对的**，确认通过。
被测对象：<repo>/scripts/results_schema.py。
全部用例不碰 GPU、不写 perf/ 目录（只在 tmp_path 里造文件）。
"""
import copy
import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "results_schema.py"


def _load():
    spec = importlib.util.spec_from_file_location(
        "_results_schema_under_test", str(SCRIPT))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


rs = _load()


def _rec(**overrides):
    """一条合法 record；overrides 直接改 placement 里的字段。"""
    rec = {
        "variant": "lightop-fused",
        "median_us": 207.4,
        "placement": {
            "grid": [128, 1],
            "ctas_per_cu": 4,
            "vgpr": 96,
            "lds_per_cta": 16384,
            "waves_per_cu": 16,
        },
        "placement_unknown": {},
    }
    for k, v in overrides.items():
        if k == "_drop":
            for f in v:
                rec["placement"].pop(f, None)
        elif k == "_unknown":
            rec["placement_unknown"] = v
        elif k == "_placement":
            rec["placement"] = v
        else:
            rec["placement"][k] = v
    return rec


def _doc(*records):
    return {"card": "fusion-demo", "records": list(records)}


def _errors(rec):
    probs, _ = rs.validate_record(rec, 0)
    return [p for p in probs if p.severity == "error"]


def _warns(rec):
    probs, _ = rs.validate_record(rec, 0)
    return [p for p in probs if p.severity == "warning"]


def _write(tmp_path, doc, name="perf.json"):
    p = tmp_path / name
    p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return p


def _run_cli(args):
    return subprocess.run([sys.executable, "-I", str(SCRIPT)] + args,
                          capture_output=True, text=True)


class TestValid(unittest.TestCase):
    def test_complete_placement_passes(self):
        """合法完整 → 0 error 0 warning；grid 列表取积。"""
        probs, info = rs.validate_record(_rec(), 0)
        self.assertEqual([], probs)
        self.assertEqual(128, info["grid_ctas"])
        self.assertEqual([], info["unknown"])

    def test_lds_zero_is_legal(self):
        """lds_per_cta=0 是真实值（不用 LDS），不是『没填』。"""
        self.assertEqual([], _errors(_rec(lds_per_cta=0)))

    def test_integer_grid_equals_list_product(self):
        """grid 写整数 128 与写 [128,1] 等价。"""
        self.assertEqual([], _errors(_rec(grid=128)))


class TestMissingField(unittest.TestCase):
    def test_missing_vgpr_is_error(self):
        """缺 vgpr 键 → error，且定位到 records[i].placement.vgpr。"""
        errs = _errors(_rec(_drop=["vgpr"]))
        self.assertTrue(errs, "缺 vgpr 却通过了")
        self.assertTrue(any("vgpr" in e.where for e in errs), errs)

    def test_vgpr_zero_is_error_not_missing(self):
        """vgpr=0 → error（0 会让 VGPR*threads=0，占用率变无限 CTA）。"""
        errs = _errors(_rec(vgpr=0))
        self.assertTrue(errs, "vgpr=0 被当成『没填』放过了")
        self.assertTrue(any("vgpr=0" in e.message for e in errs), errs)

    def test_vgpr_null_needs_reason(self):
        """null 无原因 = 与漏填无法区分 → error；带原因 → unknown。"""
        self.assertTrue(_errors(_rec(vgpr=None)))
        probs, info = rs.validate_record(_rec(vgpr=None, _unknown={
            "vgpr": "编译产物已丢失，待重编"}), 0)
        self.assertEqual([], [p for p in probs if p.severity == "error"], probs)
        self.assertIn("vgpr", info["unknown"])

    def test_placement_null_is_error(self):
        self.assertTrue(_errors(_rec(_placement=None)))

    def test_placement_missing_is_error(self):
        rec = _rec()
        del rec["placement"]
        self.assertTrue(_errors(rec), "整个 placement 缺失却被放过")


class TestUnknownField(unittest.TestCase):
    def test_unknown_placement_key_is_error(self):
        """拼写错误（vgprs）必须报错，不许静默忽略。"""
        errs = _errors(_rec(vgprs=96))
        self.assertTrue(any("vgprs" in e.where for e in errs), errs)

    def test_unknown_reason_key_is_error(self):
        rec = _rec()
        rec["placement_unknown"] = {"vgprs": "typo"}
        self.assertTrue(_errors(rec))


class TestConsistency(unittest.TestCase):
    def test_ctas_per_cu_over_cu_count_warns(self):
        """ctas_per_cu > 64 CU → warning（单 CU 不可能驻留 64 个以上）。"""
        self.assertTrue(any("num_cu" in w.message for w in _warns(
            _rec(ctas_per_cu=128))))

    def test_waves_per_cu_must_divide_ctas(self):
        """waves=15、ctas=4 → 不是整数倍 → warning。"""
        self.assertTrue(any("整数倍" in w.message
                            for w in _warns(_rec(waves_per_cu=15))))

    def test_grid_not_multiple_of_cu_warns(self):
        """hbird 反例：136 CTA 非 64 倍数 → 不均衡 warning；128 → 不警告。"""
        self.assertTrue(any("整数倍" in w.message
                            for w in _warns(_rec(grid=136))))
        self.assertFalse(any("整数倍" in w.message
                             for w in _warns(_rec(grid=128))))


class TestFillFromSo(unittest.TestCase):
    def test_fill_record_uses_real_values_keeps_grid_unknown(self):
        """VGPR/LDS/承留/waves 从 .so 行填；grid 永远 null + 原因。"""
        row = {"name": "k", "vgpr": 96, "lds_used": 8192, "max_wg": 256,
               "threads": 256, "cta_per_cu": 4}
        rec = {"variant": "v", "placement_unknown": {}}
        rp, _ = rs.fill_record(rec, row, threads=256, wave_size=64)
        self.assertEqual([], [p for p in rp if p.severity == "error"], rp)
        self.assertEqual(96, rec["placement"]["vgpr"])
        self.assertEqual(4, rec["placement"]["ctas_per_cu"])
        self.assertEqual(16, rec["placement"]["waves_per_cu"])
        self.assertIsNone(rec["placement"]["grid"])
        self.assertIn("grid", rec["placement_unknown"])

    def test_fill_record_refuses_implausible_cta_per_cu(self):
        """inspect_so 的上限 384 > 64 CU → 不写进 ctas_per_cu，写 null + 原因。"""
        row = {"name": "k", "vgpr": 2, "lds_used": 0, "max_wg": 256,
               "threads": 256, "cta_per_cu": 384}
        rec = {"variant": "v", "placement_unknown": {}}
        rs.fill_record(rec, row, threads=256, wave_size=64)
        self.assertIsNone(rec["placement"]["ctas_per_cu"])
        self.assertIn("ctas_per_cu", rec["placement_unknown"])

    def test_fill_record_without_threads_does_not_guess(self):
        """不给 --threads 时不用 max_flat_workgroup_size 硬算占用率。"""
        row = {"name": "k", "vgpr": 96, "lds_used": 8192, "max_wg": 256,
               "threads": 256, "cta_per_cu": 4}
        rec = {"variant": "v", "placement_unknown": {}}
        rs.fill_record(rec, row, threads=None, wave_size=None)
        self.assertEqual(96, rec["placement"]["vgpr"])
        self.assertIsNone(rec["placement"]["ctas_per_cu"])
        self.assertIn("threads", rec["placement_unknown"]["ctas_per_cu"])

    def test_fill_record_never_nulls_human_grid_even_with_force(self):
        """grid 是人填量：--force 也不许把它抹成 null。"""
        row = {"name": "k", "vgpr": 96, "lds_used": 8192, "max_wg": 256,
               "threads": 256, "cta_per_cu": 4}
        rec = _rec()
        rs.fill_record(rec, row, threads=256, wave_size=64, force=True)
        self.assertEqual([128, 1], rec["placement"]["grid"])
        self.assertNotIn("grid", rec["placement_unknown"])


class TestCli(unittest.TestCase):
    def test_check_rejects_missing_vgpr_exit_code(self):
        """端到端：删掉 vgpr 写进临时文件，CLI 必须非 0 退出并点名。"""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            doc = _doc(_rec(_drop=["vgpr"]))
            p = Path(td) / "bad.json"
            p.write_text(json.dumps(doc), encoding="utf-8")
            r = _run_cli(["--check", str(p)])
            self.assertNotEqual(0, r.returncode, r.stdout)
            self.assertIn("vgpr", r.stdout)

    def test_check_accepts_valid_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ok.json"
            p.write_text(json.dumps(_doc(_rec(), _rec())), encoding="utf-8")
            r = _run_cli(["--check", str(p)])
            self.assertEqual(0, r.returncode, r.stdout + r.stderr)

    def test_check_strict_turns_warning_into_failure(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "warn.json"
            p.write_text(json.dumps(_doc(_rec(grid=136))), encoding="utf-8")
            self.assertEqual(0, _run_cli(["--check", str(p)]).returncode)
            self.assertNotEqual(0, _run_cli(["--check", "--strict",
                                             str(p)]).returncode)
            # 选项在文件后也必须同样成立（nargs='+' 曾把 --strict 当成文件名）
            self.assertNotEqual(0, _run_cli(["--check", str(p),
                                             "--strict"]).returncode)

    def test_missing_file_exits_2_not_1(self):
        """文件不存在是用法错误（2），不混进『校验不通过』（1）。"""
        r = _run_cli(["--check", "/nonexistent/results.json"])
        self.assertEqual(2, r.returncode, r.stdout + r.stderr)

    def test_selftest_passes(self):
        r = _run_cli(["--selftest"])
        self.assertEqual(0, r.returncode, r.stdout + r.stderr)
        self.assertIn("selftest PASS", r.stdout)


if __name__ == "__main__":
    unittest.main()
