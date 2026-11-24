#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""transform 必须把「诊断」落到「具体改法」—— C1 验收。

背景：surfaces.yaml 的 17 条攻击面原本 transform 全为 0。
occupancy 只说「CTA/CU 太低、收益 ≤15%」，不说降 VGPR 是改 tile / unroll /
mma 形状，改成多少。这个检查器把「门」变成机器判据。

本测试既跑内置 selftest，也在真实 surfaces.yaml 上跑 CLI，
并覆盖反向判据（散文 legality 不可解析、空 transform 报错等）。
"""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts/check_transform.py"
SURFACES = REPO / "Plugins/ops/skills/op-optimize/references/surfaces.yaml"


def _load():
    spec = importlib.util.spec_from_file_location("_check_transform", str(SCRIPT))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _param():
    return {
        "kind": "param",
        "what": "{BLOCK_M, num_warps, num_stages}",
        "candidates": ["BLOCK_M 128→64", "num_warps 8→4", "num_stages 3→2"],
        "legality": "LDS = BLOCK_M*BLOCK_K*2B <= 64KB; VGPR <= 255",
        "predicted_dir": "降 VGPR ⇒ CTA/CU 从 1 升到 2 ⇒ 延迟隐藏改善",
    }


def _code():
    return {
        "kind": "code",
        "what": "累加器 fp32 分片累加",
        "legality": "rel_l2 <= 1e-3（contract 容差）",
        "predicted_dir": "寄存器压力下降 ⇒ 驻留提升",
    }


def _isa():
    return {
        "kind": "isa",
        "what": "写回从 dwordx2 提到 dwordx4",
        "legality": "N % 8 == 0; 地址 16B 对齐",
        "predicted_dir": "写带宽提升",
    }


def _surfaces(total=17, n_param=10):
    out = []
    for i in range(total):
        if i < n_param:
            t = [_param()]
        elif i < n_param + 4:
            t = [_code()]
        else:
            t = [_isa()]
        out.append({"id": "s%d" % i, "transform": t})
    return out


def _write(surfaces):
    import yaml
    d = Path(tempfile.mkdtemp())
    p = d / "surfaces.yaml"
    p.write_text(yaml.safe_dump(surfaces, allow_unicode=True, sort_keys=False),
                 encoding="utf-8")
    return p


def _run(*args, path=None):
    cmd = [sys.executable, "-I", str(SCRIPT)]
    if path is not None:
        cmd += ["--file", str(path)]
    cmd += list(args)
    return subprocess.run(cmd, capture_output=True, text=True)


class TestCheckTransform(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load()

    # ---- 内置 selftest -------------------------------------------------
    def test_selftest_passes(self):
        self.assertEqual(self.m.selftest(), 0)

    # ---- 正向：全覆盖 ------------------------------------------------
    def test_full_coverage_passes(self):
        rep = self.m.check_surfaces(_surfaces())
        self.assertTrue(rep["ok"], rep["errors"])
        self.assertEqual(rep["surface_coverage"], {"covered": 17, "total": 17})
        self.assertEqual(rep["kind_counts"]["param"], 10)

    def test_cli_full_coverage_passes(self):
        r = _run(path=_write(_surfaces()))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("17/17", r.stdout)

    # ---- 反向：缺 transform ------------------------------------------
    def test_missing_transform_reports_surface_id(self):
        s = _surfaces()
        s[3]["id"] = "occupancy"
        del s[3]["transform"]
        rep = self.m.check_surfaces(s)
        self.assertFalse(rep["ok"])
        self.assertTrue(
            any("occupancy" in e and "transform 缺失" in e for e in rep["errors"]),
            rep["errors"])

    # ---- 反向：空 transform ------------------------------------------
    def test_empty_transform_list_fails(self):
        s = _surfaces()
        s[4]["transform"] = []
        rep = self.m.check_surfaces(s)
        self.assertFalse(rep["ok"])
        self.assertTrue(any("空 list" in e for e in rep["errors"]), rep["errors"])

    # ---- 反向：kind 非法 ---------------------------------------------
    def test_invalid_kind_fails(self):
        s = _surfaces()
        s[0]["transform"][0]["kind"] = "text"
        rep = self.m.check_surfaces(s)
        self.assertFalse(rep["ok"])
        self.assertTrue(
            any("非法" in e and "text" in e for e in rep["errors"]),
            rep["errors"])

    # ---- 反向：param < 10（C1 核心门槛）------------------------------
    def test_param_below_ten_fails(self):
        rep = self.m.check_surfaces(_surfaces(n_param=9))
        self.assertFalse(rep["ok"])
        self.assertFalse(rep["param_ok"])
        self.assertTrue(any("≥10" in e for e in rep["errors"]), rep["errors"])

    # ---- 反向：param 缺 candidates -----------------------------------
    def test_param_missing_candidates_fails(self):
        s = _surfaces()
        s[0]["transform"][0].pop("candidates")
        rep = self.m.check_surfaces(s)
        self.assertFalse(rep["ok"])
        self.assertTrue(any("candidates" in e for e in rep["errors"]),
                        rep["errors"])

    # ---- 反向：param candidates 是散文 -------------------------------
    def test_param_prose_candidates_fails(self):
        s = _surfaces()
        s[0]["transform"][0]["candidates"] = ["把 tile 调小一点"]
        rep = self.m.check_surfaces(s)
        self.assertFalse(rep["ok"])
        self.assertTrue(any("不是参数替换形式" in e for e in rep["errors"]),
                        rep["errors"])

    # ---- 反向：缺 predicted_dir --------------------------------------
    def test_missing_predicted_dir_fails(self):
        s = _surfaces()
        s[0]["transform"][0].pop("predicted_dir")
        rep = self.m.check_surfaces(s)
        self.assertFalse(rep["ok"])
        self.assertTrue(any("predicted_dir" in e for e in rep["errors"]),
                        rep["errors"])

    # ---- 反向：legality 散文 → 不可解析 ------------------------------
    def test_prose_legality_unparseable(self):
        self.assertFalse(self.m.legality_parseable("注意别溢出"))
        self.assertFalse(self.m.legality_parseable("别让寄存器爆掉"))
        s = _surfaces()
        s[0]["transform"][0]["legality"] = "注意别溢出"
        rep = self.m.check_surfaces(s)
        self.assertFalse(rep["ok"])
        self.assertTrue(any("不可解析" in e for e in rep["errors"]),
                        rep["errors"])

    # ---- 反向：legality 含数值约束 → 可解析 --------------------------
    def test_numeric_legality_parseable(self):
        for v in ("LDS <= 64KB", "VGPR <= 255", "N % 8 == 0",
                  "地址 16B 对齐", "num_stages 2~4", "BLOCK_M ∈ {32, 64, 128}"):
            self.assertTrue(self.m.legality_parseable(v), v)

    # ---- 未填完不崩：报 5/17 -----------------------------------------
    def test_partial_coverage_no_crash(self):
        s = _surfaces()
        for item in s[5:]:
            del item["transform"]
        rep = self.m.check_surfaces(s)
        self.assertFalse(rep["ok"])
        self.assertEqual(rep["surface_coverage"], {"covered": 5, "total": 17})
        # CLI 也必须正常返回而不是 traceback
        r = _run(path=_write(s))
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("5/17", r.stdout)

    # ---- 空文件不崩 --------------------------------------------------
    def test_empty_surfaces_no_crash(self):
        rep = self.m.check_surfaces([])
        self.assertFalse(rep["ok"])
        self.assertEqual(rep["surface_coverage"], {"covered": 0, "total": 0})

    # ---- --legality 模式只在 legality 上表态 -------------------------
    def test_legality_cli(self):
        r = _run("--legality", path=_write(_surfaces()))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("17/17", r.stdout)

    def test_legality_cli_flags_prose(self):
        s = _surfaces()
        s[0]["transform"][0]["legality"] = "注意别溢出"
        r = _run("--legality", path=_write(s))
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)

    # ---- --json 每条都带 kind/pass -----------------------------------
    def test_json_has_kind_and_pass_per_entry(self):
        r = _run("--json", path=_write(_surfaces()))
        self.assertEqual(r.returncode, 0, r.stderr)
        rep = json.loads(r.stdout)
        self.assertEqual(rep["surface_coverage"]["total"], 17)
        for surface in rep["surfaces"]:
            self.assertIn("id", surface)
            self.assertIn("pass", surface)
            for entry in surface["entries"]:
                self.assertIn("kind", entry)
                self.assertIn("pass", entry)
                self.assertTrue(entry["pass"], entry["errors"])
                self.assertIn(entry["kind"], ("param", "code", "isa"))

    def test_json_reports_failures(self):
        s = _surfaces()
        s[0]["transform"][0]["kind"] = "text"
        r = _run("--json", path=_write(s))
        self.assertEqual(r.returncode, 1)
        rep = json.loads(r.stdout)
        self.assertFalse(rep["ok"])
        failed = [e for surface in rep["surfaces"]
                  for e in surface["entries"] if not e["pass"]]
        self.assertTrue(failed)

    # ---- 真实 surfaces.yaml：可读且不崩 ------------------------------
    def test_real_surfaces_file_readable(self):
        """真实主表当前可能还没填完（例如 0/17），但必须能正常出报告。

        注意：这里**不**断言 ok —— 主线程正在填 surfaces.yaml。
        只要 CLI 不 traceback、能报出 x/17 即可。
        """
        r = _run(path=SURFACES)
        self.assertIn(r.returncode, (0, 1), r.stderr)
        self.assertIn("/17", r.stdout)


if __name__ == "__main__":
    unittest.main()
