#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""opmanifest.py 的单测（T10 / P0-1 执行身份冻结）。

被测脚本：<repo>/scripts/opmanifest.py
无卡环境可跑：只碰临时文件、sha256 与纯函数，不需要 DCU。

钉住的几条纪律：
  - 源码比产物新 = 改完没重建，必须报
  - 产物 sha256 变了必须报（不能只看 mtime）
  - env 为 unknown / unavailable(...) / <模板> 一律算「没记录」
  - glob 匹配到多个文件时报错，不许随便挑一个（tilelang 缓存里 1700+ 个 .so）
  - 工具自身异常必须失败关闭，不 fail-open
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


opman = _load("opmanifest_under_test", REPO / "scripts" / "opmanifest.py")
SPARSE_CONTRACT = REPO / "Plugins" / "ops" / "skills" / "op-sparse-mla" / "contract.yaml"

GOOD_ENV = {"gfx": "gfx938", "dtk": "dtk2604", "torch": "2.11.0",
            "python": "3.10.12", "sclk_mhz": "600", "device": "BW1100"}
ENV_KEYS = ["gfx", "dtk", "torch", "python", "sclk_mhz", "device"]
ROLES = ["source", "build", "loaded"]


def _tree(td, src_offset=-100, lib_offset=-50):
    src = Path(td) / "kernel_src.py"
    lib = Path(td) / "executable.so"
    src.write_text("__global__ void k() {}\n", encoding="utf-8")
    lib.write_bytes(b"\x7fELF" + b"0" * 64)
    now = time.time()
    os.utime(src, (now + src_offset, now + src_offset))
    os.utime(lib, (now + lib_offset, now + lib_offset))
    return src, lib


def _record(td, src, lib, env=None, out=None):
    a = types.SimpleNamespace(
        out=str(out or (Path(td) / "manifest.json")),
        artifact=["source=%s" % src, "build=%s" % lib, "loaded=%s" % lib],
        env_json=None,
        env=["%s=%s" % (k, v) for k, v in (env if env is not None else GOOD_ENV).items()],
        from_bench=None, variant=None,
        op="sparse-mla", shape="decode-s128", note="unit")
    with contextlib.redirect_stdout(io.StringIO()):
        assert opman.cmd_record(a) == 0
    return Path(a.out)


def _validate(path, roles=ROLES, keys=ENV_KEYS):
    with open(path, "r", encoding="utf-8") as f:
        return opman.validate_manifest(json.load(f), roles, keys)


class TestHappyPath(unittest.TestCase):
    def test_record_then_validate_is_clean(self):
        with tempfile.TemporaryDirectory() as td:
            src, lib = _tree(td)
            mp = _record(td, src, lib)
            man = json.loads(mp.read_text(encoding="utf-8"))
            self.assertEqual(man["schema"], opman.SCHEMA)
            self.assertEqual(man["roles"], ["build", "loaded", "source"])
            self.assertEqual(len(man["artifacts"]), 3)
            self.assertEqual(_validate(mp), [])

    def test_record_pulls_env_from_json(self):
        with tempfile.TemporaryDirectory() as td:
            src, lib = _tree(td)
            bj = Path(td) / "bench.json"
            bj.write_text(json.dumps({"env": dict(GOOD_ENV)}), encoding="utf-8")
            a = types.SimpleNamespace(
                out=str(Path(td) / "m.json"),
                artifact=["source=%s" % src, "build=%s" % lib, "loaded=%s" % lib],
                env_json=[str(bj)], env=None, from_bench=None, variant=None,
                op="sparse-mla", shape="s", note=None)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(opman.cmd_record(a), 0)
            self.assertEqual(_validate(Path(td) / "m.json"), [])

    def test_env_json_without_env_object_is_an_error(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bench.json"
            p.write_text(json.dumps({"results": {}}), encoding="utf-8")
            with self.assertRaises(ValueError):
                opman.load_env_from_json(str(p))


class TestStaleness(unittest.TestCase):
    def test_source_newer_than_build_is_flagged(self):
        """改完没重建：源码 mtime 晚于产物。"""
        with tempfile.TemporaryDirectory() as td:
            src, lib = _tree(td, src_offset=-10, lib_offset=-100)   # 源码比产物新
            probs = _validate(_record(td, src, lib))
            self.assertTrue(any("疑似改完没重建" in p for p in probs), probs)

    def test_mtime_equal_within_grace_is_ok(self):
        with tempfile.TemporaryDirectory() as td:
            src, lib = _tree(td, src_offset=-50, lib_offset=-50)
            self.assertEqual(_validate(_record(td, src, lib)), [])

    def test_content_change_is_caught_even_if_mtime_preserved(self):
        """只改内容不动 mtime（cp -p / 回滚）也必须被 sha256 抓到。"""
        with tempfile.TemporaryDirectory() as td:
            src, lib = _tree(td)
            mp = _record(td, src, lib)
            st = os.stat(src)
            src.write_text("__global__ void k() { /* edited */ }\n", encoding="utf-8")
            os.utime(src, (st.st_atime, st.st_mtime))     # 恢复 mtime，骗过时间检查
            probs = _validate(mp)
            self.assertTrue(any("sha256 与记录不符" in p for p in probs), probs)
            self.assertFalse(any("疑似改完没重建" in p for p in probs), probs)

    def test_build_artifact_replaced_is_caught(self):
        with tempfile.TemporaryDirectory() as td:
            src, lib = _tree(td)
            mp = _record(td, src, lib)
            lib.write_bytes(b"\x7fELF" + b"1" * 64)        # 换了一份产物
            probs = _validate(mp)
            self.assertTrue(any("build 的 sha256 与记录不符" in p for p in probs), probs)

    def test_deleted_artifact_is_caught(self):
        with tempfile.TemporaryDirectory() as td:
            src, lib = _tree(td)
            mp = _record(td, src, lib)
            lib.unlink()
            probs = _validate(mp)
            self.assertTrue(any("不存在了" in p for p in probs), probs)


class TestRolesAndEnv(unittest.TestCase):
    def test_missing_required_role_is_flagged(self):
        with tempfile.TemporaryDirectory() as td:
            src, lib = _tree(td)
            mp = _record(td, src, lib)
            probs = opman.validate_manifest(
                json.loads(mp.read_text(encoding="utf-8")),
                ["source", "build", "loaded", "extra"], ENV_KEYS)
            self.assertTrue(any("缺少 role=extra" in p for p in probs), probs)

    def test_placeholder_env_values_are_flagged(self):
        for bad in ("unknown", "unavailable", "unavailable(FileNotFoundError)",
                    "n/a", "", "<dtk-version>"):
            with self.subTest(bad=bad):
                env = dict(GOOD_ENV)
                env["dtk"] = bad
                with tempfile.TemporaryDirectory() as td:
                    src, lib = _tree(td)
                    probs = _validate(_record(td, src, lib, env=env))
                    self.assertTrue(any("env.dtk" in p and "占位符" in p for p in probs), probs)

    def test_missing_env_key_is_flagged(self):
        env = dict(GOOD_ENV)
        del env["sclk_mhz"]
        with tempfile.TemporaryDirectory() as td:
            src, lib = _tree(td)
            probs = _validate(_record(td, src, lib, env=env))
            self.assertTrue(any("env.sclk_mhz 缺失" in p for p in probs), probs)

    def test_is_placeholder_matrix(self):
        for v in (None, "", "unknown", "UNKNOWN", "unavailable", "unavailable(E)",
                  "n/a", "none", "-", "tbd", "<x>"):
            self.assertTrue(opman.is_placeholder(v), v)
        for v in ("gfx938", "dtk2604", "600", 600, 938, "BW1100", "gfx938:sramecc+"):
            self.assertFalse(opman.is_placeholder(v), v)


class TestFailClosed(unittest.TestCase):
    def test_not_a_dict(self):
        self.assertTrue(opman.validate_manifest(["nope"]))

    def test_empty_artifacts(self):
        self.assertTrue(opman.validate_manifest({"schema": opman.SCHEMA, "artifacts": []}))

    def test_wrong_schema(self):
        probs = opman.validate_manifest({"schema": "v0", "artifacts": [{"role": "source"}]})
        self.assertTrue(any("schema" in p for p in probs), probs)

    def test_cli_on_unreadable_manifest_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as td:
            bad = Path(td) / "manifest.json"
            bad.write_text("{not json", encoding="utf-8")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = opman.main(["validate", "--manifest", str(bad)])
            self.assertEqual(rc, 1)
            self.assertIn("执行身份未冻结", buf.getvalue())

    def test_cli_validate_ok_and_json(self):
        with tempfile.TemporaryDirectory() as td:
            src, lib = _tree(td)
            mp = _record(td, src, lib)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = opman.main(["validate", "--manifest", str(mp),
                                 "--contract", str(SPARSE_CONTRACT), "--json"])
            self.assertEqual(rc, 0, buf.getvalue())
            payload = json.loads(buf.getvalue())
            self.assertTrue(payload["ok"], payload)
            self.assertEqual(payload["required_roles"], ["source", "loaded"])


class TestGlobResolution(unittest.TestCase):
    def test_unique_glob_resolves(self):
        with tempfile.TemporaryDirectory() as td:
            src, lib = _tree(td)
            art = opman.parse_artifact_spec("build=%s/*.so" % td)
            self.assertEqual(art["path"], str(lib.resolve()))

    def test_ambiguous_glob_refuses_to_guess(self):
        """tilelang 缓存下有 1700+ 个 executable.so —— 猜必错，必须报错。"""
        with tempfile.TemporaryDirectory() as td:
            _tree(td)
            (Path(td) / "other.so").write_bytes(b"\x7fELF")
            with self.assertRaises(ValueError) as cm:
                opman.parse_artifact_spec("build=%s/*.so" % td)
            self.assertIn("必须正好 1 个", str(cm.exception))

    def test_unknown_role_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            src, _ = _tree(td)
            with self.assertRaises(ValueError):
                opman.parse_artifact_spec("binary=%s" % src)

    def test_missing_file_rejected(self):
        with self.assertRaises(ValueError):
            opman.parse_artifact_spec("source=/definitely/not/here.py")


class TestContractIntegration(unittest.TestCase):
    def test_contract_requirements_are_read(self):
        roles, keys = opman.contract_requirements(str(SPARSE_CONTRACT))
        # build 是可选的：tilelang 的 build 与 loaded 是同一个 .so
        self.assertEqual(roles, ["source", "loaded"])
        self.assertIn("sclk_mhz", keys)


class TestFromBench(unittest.TestCase):
    """`record --from-bench`：产物路径来自 bench 进程当场抓的 artifacts，不是事后猜的。"""

    def _bench(self, td, artifacts=None, env=None, variant="tilelang", drop_variant=False):
        src, lib = _tree(td)                    # 产物必须真的存在（record 会 stat 它）
        art = {"source": [str(src)], "loaded": [str(lib)],
               "loaded_from": "tilelang JITImpl._kernel_cache[].adapter.libpath"}
        if artifacts is not None:
            art = artifacts
        results = {} if drop_variant else {variant: {"status": "ok", "artifacts": art}}
        p = Path(td) / "bench.json"
        p.write_text(json.dumps({"op": "sparse-mla", "shape": "decode-glm53",
                                 "env": dict(GOOD_ENV) if env is None else env,
                                 "results": results}), encoding="utf-8")
        return p

    def _record_from_bench(self, td, bench, variant="tilelang", artifact=None):
        a = types.SimpleNamespace(
            out=str(Path(td) / "manifest.json"), artifact=artifact or [],
            env_json=[str(bench)], env=None,
            from_bench=str(bench), variant=variant,
            op="sparse-mla", shape="decode-glm53", note="unit")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(opman.cmd_record(a), 0)
        return Path(a.out)

    def test_expands_source_and_loaded(self):
        with tempfile.TemporaryDirectory() as td:
            _src, lib = _tree(td)               # 同一条 tree，路径要对得上
            mp = self._record_from_bench(td, self._bench(td))
            man = json.loads(mp.read_text(encoding="utf-8"))
            self.assertEqual(sorted(man["roles"]), ["loaded", "source"])
            paths = {x["role"]: x["path"] for x in man["artifacts"]}
            self.assertEqual(paths["loaded"], str(lib.resolve()))

    def test_missing_variant_lists_what_is_available(self):
        with tempfile.TemporaryDirectory() as td:
            b = self._bench(td, variant="flash_mla")
            with self.assertRaises(ValueError) as cm:
                opman.artifacts_from_bench(str(b), "tilelang")
            self.assertIn("flash_mla", str(cm.exception))

    def test_artifacts_absent_surfaces_loaded_note(self):
        """bench 没做身份采集时要报出它自己的原因，不能只说「没有」。"""
        with tempfile.TemporaryDirectory() as td:
            b = self._bench(td, artifacts={"loaded_note": "运行时没取到加载产物路径"})
            with self.assertRaises(ValueError) as cm:
                opman.artifacts_from_bench(str(b), "tilelang")
            self.assertIn("运行时没取到", str(cm.exception))

    def test_empty_artifacts_is_an_error(self):
        with tempfile.TemporaryDirectory() as td:
            b = self._bench(td, artifacts={"source": [], "loaded": []})
            with self.assertRaises(ValueError):
                opman.artifacts_from_bench(str(b), "tilelang")

    def test_same_artifact_given_twice_is_deduped(self):
        with tempfile.TemporaryDirectory() as td:
            bench = self._bench(td)
            _src, lib = _tree(td)
            mp = self._record_from_bench(td, bench, artifact=["loaded=%s" % lib])
            man = json.loads(mp.read_text(encoding="utf-8"))
            loaded = [x for x in man["artifacts"] if x["role"] == "loaded"]
            self.assertEqual(len(loaded), 1, loaded)

    def test_bench_env_missing_key_is_not_silent(self):
        """人工拼的 bench.json 缺 sclk_mhz：record 不会拦，但 validate 必须点名报出来。"""
        env = {k: v for k, v in GOOD_ENV.items() if k != "sclk_mhz"}
        with tempfile.TemporaryDirectory() as td:
            mp = self._record_from_bench(td, self._bench(td, env=env))
            probs = _validate(mp)
            self.assertTrue(any("env.sclk_mhz 缺失" in p for p in probs), probs)


if __name__ == "__main__":
    unittest.main()
