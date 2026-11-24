#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""兼容 shim —— 实现已迁到 `Plugins/ops/engine/loop.py`。

**本文件不含实现。** 保留 3 个迁移周期，不打断现有引用。
边界规则见 `docs/ENGINE.md`；行数与零函数定义由 test_loop_uniqueness 守住。

⚠️ 已知局限：被 import 时只拷名字，**不共享可变全局**。
测试若改 `loop.VERIFY_TIMEOUT_S` 再调函数会失效——那类测试应直连 engine。
"""
import pathlib
import runpy
import sys

_HERE = pathlib.Path(__file__).resolve()
_TARGET = _HERE.parents[2] / "engine" / "loop.py"      # parents[2] = Plugins/ops

if not _TARGET.is_file():
    sys.exit("ERROR: 找不到 %s —— ENGINE 迁移未完成" % _TARGET)

if __name__ == "__main__":
    print("[deprecated] 已迁到 engine/loop.py；本 shim 保留 3 个版本", file=sys.stderr)
    runpy.run_path(str(_TARGET), run_name="__main__")
else:
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("op_optimize_engine", str(_TARGET))
    _mod = _ilu.module_from_spec(_spec)
    sys.modules["op_optimize_engine"] = _mod
    _spec.loader.exec_module(_mod)
    globals().update(_mod.__dict__)
