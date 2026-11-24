#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kernel_scaffold 的参数化（v6 C5）测试。

这里守的不是「能生成字符串」，是**参数真的进了编译产物**：
    - BLOCK_ROWS 改 grid 与 LDS（静态 __shared__ 表进 metadata，可回读）
    - WARPS/THREADS 经 __launch_bounds__(THREADS) 进 kernel metadata
      （只改 host 侧 const int threads 不会改 metadata —— 这是「假变体」的入口）

反向用例与正向同样重要：不认识的参数/越界/自相矛盾的 THREADS 必须报错，
不许静默回退到默认值。
"""
import importlib.util
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCAFFOLD = REPO / "scripts/kernel_scaffold.py"


def _load():
    spec = importlib.util.spec_from_file_location(
        "_ks", str(SCAFFOLD))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _have_hipcc():
    return shutil.which("hipcc") is not None


class TestParamsParsing(unittest.TestCase):
    """纯解析层：不编译，跑得快。"""

    @classmethod
    def setUpClass(cls):
        cls.m = _load()

    def test_defaults_when_no_params(self):
        """不传 --params → 全默认。"""
        self.assertEqual(self.m.parse_params("reduce-lastdim", None),
                         {"BLOCK_ROWS": 1, "WARPS": 4, "THREADS": 256})
        self.assertEqual(self.m.parse_params("reduce-lastdim", ""),
                         {"BLOCK_ROWS": 1, "WARPS": 4, "THREADS": 256})

    def test_partial_params_use_defaults(self):
        """缺某个参数 → 用默认值，不报错。"""
        v = self.m.parse_params("reduce-lastdim", "BLOCK_ROWS=2")
        self.assertEqual(v, {"BLOCK_ROWS": 2, "WARPS": 4, "THREADS": 256})

    def test_full_params(self):
        v = self.m.parse_params(
            "reduce-lastdim", "BLOCK_ROWS=4,WARPS=8,THREADS=512")
        self.assertEqual(v, {"BLOCK_ROWS": 4, "WARPS": 8, "THREADS": 512})

    def test_reverse_unknown_param_lists_supported(self):
        """反向：不认识的参数名 → 报错并列出支持集合。"""
        with self.assertRaises(self.m.ParamError) as cm:
            self.m.parse_params("reduce-lastdim", "BLOCK_ROWS=2,FOO=1")
        msg = str(cm.exception)
        self.assertIn("FOO", msg)
        for name in ("BLOCK_ROWS", "WARPS", "THREADS"):
            self.assertIn(name, msg)

    def test_reverse_threads_not_warps_times_64(self):
        """反向：THREADS != WARPS*64 → 报错。"""
        with self.assertRaises(self.m.ParamError) as cm:
            self.m.parse_params("reduce-lastdim", "WARPS=4,THREADS=128")
        self.assertIn("THREADS", str(cm.exception))

    def test_reverse_block_rows_zero(self):
        """反向：越界 BLOCK_ROWS=0 → 报错，且提示范围。"""
        with self.assertRaises(self.m.ParamError) as cm:
            self.m.parse_params("reduce-lastdim", "BLOCK_ROWS=0")
        self.assertIn("1~8", str(cm.exception))

    def test_reverse_warps_hundred(self):
        """反向：越界 WARPS=100 → 报错。"""
        with self.assertRaises(self.m.ParamError):
            self.m.parse_params("reduce-lastdim", "WARPS=100")

    def test_reverse_non_integer(self):
        with self.assertRaises(self.m.ParamError):
            self.m.parse_params("reduce-lastdim", "BLOCK_ROWS=abc")

    def test_generate_reports_error_not_exception(self):
        """generate 走 err 返回，方便 CLI 统一处理。"""
        src, err = self.m.generate("reduce-lastdim", "hip", "WARPS=4,THREADS=1")
        self.assertIsNone(src)
        self.assertIn("THREADS", err)


class TestRenderedSource(unittest.TestCase):
    """生成的源码：宏真的在、真的被用。不依赖编译。"""

    @classmethod
    def setUpClass(cls):
        cls.m = _load()

    def test_backward_compatible_default_output(self):
        """不传 --params 时与旧版结构一致：仍生成能编译的默认骨架。"""
        a, err = self.m.generate("reduce-lastdim", "hip")
        self.assertIsNone(err)
        self.assertIn("#define BLOCK_ROWS 1", a)
        self.assertIn("#define WARPS 4", a)
        self.assertIn("#define THREADS 256", a)

    def test_macros_greppable_and_used(self):
        """参数值能 grep 到，且进了 kernel 结构与 launch。"""
        src, err = self.m.generate(
            "reduce-lastdim", "hip", "BLOCK_ROWS=4,WARPS=8,THREADS=512")
        self.assertIsNone(err)
        self.assertIn("#define BLOCK_ROWS 4", src)
        self.assertIn("#define WARPS 8", src)
        self.assertIn("#define THREADS 512", src)
        # THREADS 必须进 __launch_bounds__（否则编译产物 threads 不变）
        self.assertIn("__launch_bounds__(THREADS)", src)
        # BLOCK_ROWS 必须进 grid 与动态 LDS 大小
        self.assertIn("(M + BLOCK_ROWS - 1) / BLOCK_ROWS", src)
        self.assertIn("sizeof(float) * BLOCK_ROWS", src)
        # WARPS 必须决定跨 warp 归约表长度
        self.assertIn("warp_part[BLOCK_ROWS][WARPS]", src)

    def test_each_block_rows_changes_grid_expr(self):
        """BLOCK_ROWS 不同 → 源码里宏值不同（grid 表达式由宏驱动）。"""
        seen = set()
        for br in (1, 2, 4):
            src, _ = self.m.generate(
                "reduce-lastdim", "hip",
                {"BLOCK_ROWS": br, "WARPS": 4, "THREADS": 256})
            self.assertIn("#define BLOCK_ROWS %d" % br, src)
            seen.add(br)
        self.assertEqual(seen, {1, 2, 4})


@unittest.skipUnless(_have_hipcc(), "需要 hipcc")
class TestCompiledResources(unittest.TestCase):
    """真编译 + 真回读 —— 本任务的核心验收。"""

    @classmethod
    def setUpClass(cls):
        cls.m = _load()
        cls.tmp = Path(tempfile.mkdtemp())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _build(self, br, w, t):
        src, err = self.m.generate(
            "reduce-lastdim", "hip",
            {"BLOCK_ROWS": br, "WARPS": w, "THREADS": t})
        self.assertIsNone(err, err)
        f = self.tmp / ("k_%d_%d_%d.hip" % (br, w, t))
        f.write_text(src, encoding="utf-8")
        prof = self.m.build_and_inspect(f)
        self.assertTrue(prof.get("compiled"), prof)
        self.assertNotIn("error", prof, prof)
        return src, prof

    def test_generated_code_compiles(self):
        """最低标准：默认参数生成的代码能编译。"""
        _, prof = self._build(1, 4, 256)
        self.assertTrue(prof["compiled"])

    def test_four_combos_have_distinct_profiles(self):
        """核心验收：4 组参数 → 编译后 VGPR 或 LDS 互不相同。"""
        combos = [(1, 4, 256), (4, 4, 256), (1, 8, 512), (2, 2, 128)]
        profiles = [self._build(*c)[1] for c in combos]
        sigs = {(p["vgpr"], p["lds"], p["threads"]) for p in profiles}
        self.assertEqual(len(sigs), len(combos),
                         "四组参数编译后画像重复：%r" % (sigs,))

    def test_block_rows_scales_lds_4x(self):
        """BLOCK_ROWS 1→4，静态 LDS 必须按 4 倍增长（证明参数真进了 kernel）。

        warp_part[BLOCK_ROWS][WARPS] 是静态 __shared__，进 metadata 可回读。
        (动态 extern __shared__ 不进 metadata，这里刻意不拿它当判据。)
        """
        _, p1 = self._build(1, 4, 256)
        _, p4 = self._build(4, 4, 256)
        self.assertEqual(p4["lds"], p1["lds"] * 4,
                         "BLOCK_ROWS 1→4 时 LDS 未 4 倍增长：%d → %d"
                         % (p1["lds"], p4["lds"]))

    def test_warps_drives_threads_in_binary(self):
        """WARPS 变化 → threads 跟着变 → 编译产物 threads 字段跟着变。"""
        for w, t in ((2, 128), (4, 256), (8, 512)):
            _, prof = self._build(1, w, t)
            self.assertEqual(prof["threads"], w * 64,
                             "WARPS=%d 期望产物 threads=%d，读到 %d"
                             % (w, w * 64, prof["threads"]))

    def test_threads_comes_from_launch_bounds(self):
        """硬证据：产物 threads == THREADS 宏，而不是 host 侧变量。"""
        src, prof = self._build(1, 8, 512)
        self.assertIn("__launch_bounds__(THREADS)", src)
        self.assertEqual(prof["threads"], 512)

    def test_resource_profile_changes_with_parameters(self):
        """至少 VGPR 有两组不同 —— 证明编译真的受参数影响。"""
        vgprs = {self._build(*c)[1]["vgpr"]
                 for c in ((1, 4, 256), (4, 4, 256), (2, 2, 128))}
        self.assertGreater(len(vgprs), 1, "VGPR 不随参数变化：%r" % vgprs)

    def test_binary_runs(self):
        """编译产物能跑出计时，且打印出参数化的 grid/threads。"""
        _, prof = self._build(4, 4, 256)
        exe = self.tmp / "k_4_4_256"
        p = subprocess.run([str(exe), "1024", "512"], capture_output=True,
                           text=True, timeout=120)
        self.assertEqual(p.returncode, 0, p.stderr[:400])
        self.assertIn("median_ms=", p.stdout)
        self.assertIn("grid=256", p.stdout)   # M=1024 / BLOCK_ROWS=4
        self.assertIn("threads=256", p.stdout)


if __name__ == "__main__":
    unittest.main()
