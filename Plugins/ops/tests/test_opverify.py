#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""opverify.py 的单测：契约缺字段 → REJECTED、四档结论、不变量失败 → 不通过。

被测脚本：<repo>/scripts/opverify.py（共享脚本，本目录只测不 workload）。
无卡环境可跑：全部用例只碰文本/YAML/JSON 与纯函数，不需要 DCU。
"""
import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


opverify = _load("opverify_under_test", REPO / "scripts" / "opverify.py")
opman = _load("opmanifest_under_test", REPO / "scripts" / "opmanifest.py")

SPARSE_CONTRACT = REPO / "Plugins" / "ops" / "skills" / "op-sparse-mla" / "contract.yaml"

GOOD_ENV = ["gfx=gfx938", "dtk=dtk2604", "torch=2.11.0", "python=3.10.12",
            "sclk_mhz=600", "device=BW1100"]


def _sparse_run():
    return {
        "op": "sparse-mla", "shape": "decode-s128", "variant": "tilelang",
        "env": {"gfx": "gfx938", "dtk": "dtk2604"},
        "correctness": {"max_abs": 6.8e-4, "cosine": 0.999998, "n_compared": 1024},
        "invariants": {"determinism": True, "index_permutation_invariance": True,
                       "split_invariance": True},
        "metrics": {"median_ms": 1.00, "mad_ms": 0.01},
        "baseline": {"median_ms": 1.10, "mad_ms": 0.02},
    }


def _record_manifest(td, src, lib, out=None, env=None):
    """按真实命令路径录一份 manifest（不走捷径，测的就是 record 本身）。"""
    a = types.SimpleNamespace(
        out=str(out or (Path(td) / "manifest.json")),
        artifact=["source=%s" % src, "build=%s" % lib, "loaded=%s" % lib],
        env_json=None, env=list(env if env is not None else GOOD_ENV),
        from_bench=None, variant=None,
        op="sparse-mla", shape="decode-s128", note="unit test")
    with contextlib.redirect_stdout(io.StringIO()):
        rc = opman.cmd_record(a)
    assert rc == 0, rc
    return Path(a.out)


def _make_tree(td):
    """造一条合法的 source/build 时间链：源码先写，产物后写。"""
    src = Path(td) / "kernel_src.py"
    lib = Path(td) / "executable.so"
    src.write_text("__global__ void k() {}\n", encoding="utf-8")
    lib.write_bytes(b"\x7fELF" + b"0" * 64)
    base = time.time() - 100
    os.utime(src, (base, base))
    os.utime(lib, (base + 50, base + 50))
    return src, lib


def _contract(**over):
    c = {
        "op": "demo",
        "shapes": {"x": ["M", "K"]},
        "dtypes": {"x": "bfloat16", "acc": "float32"},
        "tolerance": {
            "vs_fp32_ref": {"max_abs": 2.0e-3, "cosine": 0.9999},
            "variant_vs_variant": "bit_exact",
        },
        "invariants": ["determinism"],
        "gates": [{"name": "perf", "require": "median_ms <= baseline_ms * 1.02"}],
        "floor": {"metric": "hbm_bandwidth", "peak": 1724.0, "unit": "GB/s",
                  "basis": "每 token 读 X 字节 / 峰值带宽", "at_floor_threshold": 0.85},
    }
    c.update(over)
    return c


def _run(**over):
    r = {
        "op": "demo",
        "shape": "decode-s128",
        "variant": "tilelang",
        "env": {"gfx": "gfx938", "dtk": "dtk2604"},
        "correctness": {"max_abs": 6.8e-4, "cosine": 0.999998, "n_compared": 1024},
        "invariants": {"determinism": True},
        "metrics": {"median_ms": 1.00, "mad_ms": 0.01},
        "baseline": {"median_ms": 1.10, "mad_ms": 0.02},
    }
    r.update(over)
    return r


class TestContractCheck(unittest.TestCase):
    def test_repo_sample_contract_is_clean(self):
        c = opverify.parse_yaml(SPARSE_CONTRACT.read_text(encoding="utf-8"))
        self.assertEqual(c["op"], "sparse-mla")
        self.assertEqual(c["shapes"]["q"], ["S", "H", 576])
        self.assertEqual(c["invariants"][0]["name"], "determinism")
        self.assertAlmostEqual(c["floor"]["at_floor_threshold"], 0.85)
        self.assertEqual(opverify.check_contract(c, str(SPARSE_CONTRACT.parent)), [])

    def test_missing_required_fields_are_reported(self):
        incomplete = {"op": "demo", "shapes": {"x": ["M", "K"]}, "dtypes": {"x": "bfloat16"}}
        probs = opverify.check_contract(incomplete)
        joined = " | ".join(probs)
        for field in ("tolerance", "invariants", "gates", "floor"):
            self.assertIn("缺少必填字段: %s" % field, joined)
        self.assertIn("tolerance.vs_fp32_ref.max_abs 缺失", joined)

    def test_contract_missing_field_makes_cli_reject(self):
        """CI 口径：契约缺字段时 --run 也没用，必须 REJECTED 且退出码非零。"""
        text = "op: demo\nshapes:\n  x: [M, K]\ndtypes:\n  x: bfloat16\n"
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "contract.yaml"
            p.write_text(text, encoding="utf-8")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = opverify.main(["--contract", str(p), "--json"])
            payload = json.loads(buf.getvalue())
        self.assertEqual(rc, 1)
        self.assertEqual(payload["verdict"], "REJECTED")
        self.assertTrue(payload["missing"])

    def test_floor_basis_cannot_be_empty(self):
        c = _contract(floor={"metric": "hbm_bandwidth", "peak": 1724.0,
                             "unit": "GB/s", "basis": "", "at_floor_threshold": 0.85})
        probs = opverify.check_contract(c)
        self.assertTrue(any("floor.basis" in x for x in probs), probs)

    def test_required_gate_without_script_is_a_problem(self):
        c = _contract(gates=[{"name": "correctness", "required": True}])
        probs = opverify.check_contract(c)
        self.assertTrue(any("required=true 但没有 script" in x for x in probs), probs)


class TestVerdicts(unittest.TestCase):
    """四档结论各一条（ACCEPTED / REJECTED / INCONCLUSIVE / DIAGNOSTIC-ONLY）。"""

    def test_accepted(self):
        v = opverify.evaluate(_contract(), _run())
        self.assertEqual(v["verdict"], "ACCEPTED", v)
        self.assertEqual(v["gates"], {"correctness": "pass", "invariants": "pass", "perf": "pass"})
        self.assertEqual(v["missing"], [])

    def test_rejected_on_invariant_failure(self):
        run = _run(invariants={"determinism": False})
        v = opverify.evaluate(_contract(), run)
        self.assertEqual(v["verdict"], "REJECTED", v)
        self.assertEqual(v["gates"]["invariants"], "fail")
        self.assertEqual(v["gates"]["correctness"], "pass")
        self.assertTrue(any("不变量失败" in r for r in v["reasons"]))

    def test_inconclusive_when_within_noise(self):
        run = _run(metrics={"median_ms": 1.09, "mad_ms": 0.02})   # Δ=0.01 < 3*MAD=0.06
        v = opverify.evaluate(_contract(), run)
        self.assertEqual(v["verdict"], "INCONCLUSIVE", v)
        self.assertEqual(v["gates"]["perf"], "inconclusive")

    def test_diagnostic_only_when_perf_allowed_missing(self):
        run = _run()
        del run["metrics"]
        v = opverify.evaluate(_contract(), run, allow_missing=["perf"])
        self.assertEqual(v["verdict"], "DIAGNOSTIC-ONLY", v)
        self.assertEqual(v["gates"]["perf"], "skipped")

    def test_invariant_not_measured_is_rejected(self):
        run = _run(invariants={})
        v = opverify.evaluate(_contract(), run)
        self.assertEqual(v["verdict"], "REJECTED", v)
        self.assertTrue(any("invariant.determinism" in m for m in v["missing"]), v["missing"])

    def test_out_of_tolerance_is_rejected(self):
        run = _run(correctness={"max_abs": 5.0e-3, "cosine": 0.99999})
        v = opverify.evaluate(_contract(), run)
        self.assertEqual(v["verdict"], "REJECTED", v)
        self.assertEqual(v["gates"]["correctness"], "fail")

    def test_cosine_below_limit_is_rejected(self):
        run = _run(correctness={"max_abs": 1.0e-4, "cosine": 0.99})
        v = opverify.evaluate(_contract(), run)
        self.assertEqual(v["verdict"], "REJECTED", v)

    def test_missing_perf_data_is_rejected_not_default_pass(self):
        run = _run()
        del run["baseline"]
        v = opverify.evaluate(_contract(), run)
        self.assertEqual(v["verdict"], "REJECTED", v)
        self.assertTrue(any("baseline.median_ms" in m for m in v["missing"]))

    def test_missing_env_fingerprint_is_rejected(self):
        run = _run(env={"gfx": "gfx938"})
        v = opverify.evaluate(_contract(), run)
        self.assertEqual(v["verdict"], "REJECTED", v)
        self.assertTrue(any("env.dtk" in m for m in v["missing"]))
        # 显式关掉指纹检查后不再因此拒绝
        v2 = opverify.evaluate(_contract(), run, env_check=False)
        self.assertEqual(v2["verdict"], "ACCEPTED", v2)

    def test_op_mismatch_is_rejected(self):
        run = _run(op="something-else")
        v = opverify.evaluate(_contract(), run)
        self.assertEqual(v["verdict"], "REJECTED", v)
        self.assertTrue(any("run.op" in m for m in v["missing"]))


class TestCliIntegration(unittest.TestCase):
    def test_real_contract_plus_valid_run_is_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            src, lib = _make_tree(td)
            _record_manifest(td, src, lib)
            rp = Path(td) / "result.json"
            rp.write_text(json.dumps(_sparse_run()), encoding="utf-8")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = opverify.main(["--contract", str(SPARSE_CONTRACT), "--run", str(rp), "--json"])
            payload = json.loads(buf.getvalue())
        self.assertEqual(rc, 0, payload)
        self.assertEqual(payload["verdict"], "ACCEPTED", payload)
        self.assertEqual(payload["op"], "sparse-mla")
        self.assertEqual(payload["gates"]["identity"], "pass", payload)

    def test_real_contract_without_manifest_is_rejected(self):
        """P0-1：真实契约要求 manifest，没有 manifest 必须 REJECTED —— 数字配不了代码就没意义。"""
        with tempfile.TemporaryDirectory() as td:
            rp = Path(td) / "result.json"
            rp.write_text(json.dumps(_sparse_run()), encoding="utf-8")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = opverify.main(["--contract", str(SPARSE_CONTRACT), "--run", str(rp), "--json"])
            payload = json.loads(buf.getvalue())
        self.assertEqual(rc, 1, payload)
        self.assertEqual(payload["verdict"], "REJECTED", payload)
        self.assertEqual(payload["gates"]["identity"], "fail", payload)
        self.assertTrue(any("执行身份未冻结" in r for r in payload["reasons"]), payload["reasons"])

    def test_source_edited_without_rebuild_is_rejected(self):
        """P0-1 的核心验收：改一行源码不重建 -> REJECTED，理由指到具体文件。"""
        with tempfile.TemporaryDirectory() as td:
            src, lib = _make_tree(td)
            mp = _record_manifest(td, src, lib)
            # 改源码、不重建：内容变 + mtime 超过产物
            src.write_text("__global__ void k() { /* edit */ }\n", encoding="utf-8")
            t = time.time() - 40          # 仍早于「现在」，但晚于 base+50 的产物
            os.utime(src, (t, t))
            rp = Path(td) / "result.json"
            rp.write_text(json.dumps(_sparse_run()), encoding="utf-8")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = opverify.main(["--contract", str(SPARSE_CONTRACT), "--run", str(rp),
                                    "--manifest", str(mp), "--json"])
            payload = json.loads(buf.getvalue())
        self.assertEqual(rc, 1, payload)
        self.assertEqual(payload["verdict"], "REJECTED", payload)
        reasons = " | ".join(payload["reasons"])
        self.assertIn("sha256 与记录不符", reasons)
        self.assertIn("疑似改完没重建", reasons)
        self.assertIn("kernel_src.py", reasons)

    def test_explicit_missing_manifest_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            rp = Path(td) / "result.json"
            rp.write_text(json.dumps(_sparse_run()), encoding="utf-8")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = opverify.main(["--contract", str(SPARSE_CONTRACT), "--run", str(rp),
                                    "--manifest", str(Path(td) / "nope.json"), "--json"])
            payload = json.loads(buf.getvalue())
        self.assertEqual(rc, 1, payload)
        self.assertTrue(any("不存在" in r for r in payload["reasons"]), payload["reasons"])

    def test_allow_missing_manifest_downgrades_to_skip(self):
        """显式逃生口：允许跳过身份校验，但闸上必须留痕（skipped），不能悄悄 pass。"""
        with tempfile.TemporaryDirectory() as td:
            rp = Path(td) / "result.json"
            rp.write_text(json.dumps(_sparse_run()), encoding="utf-8")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = opverify.main(["--contract", str(SPARSE_CONTRACT), "--run", str(rp),
                                    "--allow-missing", "manifest", "--json"])
            payload = json.loads(buf.getvalue())
        self.assertEqual(payload["gates"]["identity"], "skipped", payload)
        self.assertTrue(any("显式跳过" in r for r in payload["reasons"]), payload["reasons"])

    def test_check_contract_mode_reports_ok(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = opverify.main(["--contract", str(SPARSE_CONTRACT), "--check-contract"])
        self.assertEqual(rc, 0)
        self.assertIn("契约合格", buf.getvalue())

    def test_contract_rejects_manifest_without_build_role(self):
        c = opverify.parse_yaml(SPARSE_CONTRACT.read_text(encoding="utf-8"))
        c["manifest"] = {"required_roles": ["source"], "env_keys": ["gfx"]}
        probs = opverify.check_contract(c)
        self.assertTrue(any("至少要含 build 或 loaded" in x for x in probs), probs)

    def test_contract_rejects_unknown_role(self):
        c = opverify.parse_yaml(SPARSE_CONTRACT.read_text(encoding="utf-8"))
        c["manifest"] = {"required_roles": ["source", "binary"], "env_keys": ["gfx"]}
        probs = opverify.check_contract(c)
        self.assertTrue(any("未知 role" in x for x in probs), probs)


if __name__ == "__main__":
    unittest.main()
