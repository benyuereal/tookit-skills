#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""op-roofline 机器模型的常数来源守卫。

为什么要这个测试：`models/*.py` 曾经把 CU 数写死成 69，而 constants.yaml 早已
更正为 64（2026-11-13）。写死 69 会让所有按 CU 归一的量（per-CU 摊销、rounds、
waves_per_cu、MFMA/CU）偏 ~8%，方向是「以为还能再塞一点」，而实际已到顶 ——
优化结论会反。

本测试只守一件事：**模型里的机器常数必须来自设备查询或 constants.yaml，
不许回退成写死的字面量**。无卡环境可跑（会走 fallback 路径）。
"""
import importlib.util
import io
import re
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MODELS = REPO / "Plugins/ops/skills/op-roofline/models"
CONSTANTS = REPO / "Plugins/ops/skills/op-roofline/constants.yaml"


def _load(name):
    path = MODELS / ("%s.py" % name)
    spec = importlib.util.spec_from_file_location("_model_" + name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _constant_value(key):
    """从 constants.yaml 读 compute.<key>.value（避免依赖 pyyaml）。"""
    text = CONSTANTS.read_text(encoding="utf-8")
    m = re.search(r"(?m)^\s{2}%s:\s*$\n\s{4}value:\s*(\d+)" % re.escape(key), text)
    return int(m.group(1)) if m else None


class TestModelConstants(unittest.TestCase):
    def test_cuda_cu_count_matches_constants_yaml(self):
        """CU 数必须与 constants.yaml 的实测值一致（写死 69 是历史错误）。"""
        declared = _constant_value("cu_count")
        self.assertIsNotNone(declared, "constants.yaml 缺 compute.cu_count")
        for name in ("sparse_mla", "mqa_logits"):
            mod = _load(name)
            with redirect_stderr(io.StringIO()):
                cu = mod.CU
            self.assertEqual(
                cu, declared,
                "%s.CU=%s 与 constants.yaml 的 %s 不一致；它应来自设备查询，"
                "读不到时退回 constants.yaml 的值" % (name, cu, declared))

    def test_cu_fallback_is_not_the_old_wrong_value(self):
        """fallback 不许再是 69。69 来自 hy-smi 逐卡读数，已被三方否证。"""
        for name in ("sparse_mla", "mqa_logits"):
            mod = _load(name)
            self.assertNotEqual(
                getattr(mod, "CU_FALLBACK", None), 69,
                "%s.CU_FALLBACK 仍是已被否证的 69" % name)

    def test_models_expose_device_query(self):
        """两个模型都必须有 _cu_count，供设备查询与 fallback 统一走一条路。"""
        for name in ("sparse_mla", "mqa_logits"):
            mod = _load(name)
            self.assertTrue(callable(getattr(mod, "_cu_count", None)),
                            "%s 缺少 _cu_count()" % name)

    def test_no_bare_cu_literal_in_models(self):
        """模型源码里不许再出现 `CU = <数字>` 这种写死写法。"""
        bad = []
        for path in MODELS.glob("*.py"):
            for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if re.match(r"^\s*CU\s*=\s*\d+\s*(#.*)?$", line):
                    bad.append("%s:%d: %s" % (path.name, i, line.strip()))
        self.assertEqual(bad, [], "发现写死的 CU 字面量：\n" + "\n".join(bad))


if __name__ == "__main__":
    unittest.main()
