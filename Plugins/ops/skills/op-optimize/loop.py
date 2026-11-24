#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""兼容 shim —— 实现已迁到 `Plugins/ops/engine/loop.py`。

**本文件不含任何实现。** 它只做转发，保留 3 个迁移周期，用来不打断
现有的 51 处引用（5 处 importlib 测试、40+ 处文档路径、1 处跨插件
`dcu-orchestrator/tasks/op-optimize.yaml`）。

新代码请直接用 engine 版：
    python3 -I Plugins/ops/engine/loop.py -C <卡> <子命令>

为什么迁：`engine/loop.py` 有 1942 行、12 个子命令，被另一个插件引用。
它已经是「调度引擎」而不是 op-optimize 这个 skill 的实现细节。
边界规则见 `docs/ENGINE.md`。

判据：本文件行数必须 ≤ 30，且不许出现 `def cmd_`（由
`tests/test_loop_uniqueness.py` 守住）。超了说明有人往 shim 里写实现。
"""
import pathlib
import runpy
import sys

_HERE = pathlib.Path(__file__).resolve()
# parents[2] = Plugins/ops（parents[3] 是 Plugins，会算错）
_TARGET = _HERE.parents[2] / "engine" / "loop.py"

if not _TARGET.is_file():
    sys.exit("ERROR: 找不到 %s —— ENGINE 迁移未完成，实现可能还在别处" % _TARGET)

print("[deprecated] skills/op-optimize/loop.py 已迁到 engine/loop.py；"
      "本 shim 保留 3 个版本", file=sys.stderr)

# 只有在被**当作脚本执行**时才转发。
# 测试会用 importlib 加载本模块（拿到 cmd_* 函数），那种情况下必须只做
# 「把 engine 的名字空间倒进来」，不能执行 main()——否则 argparse 在
# 无参数时 sys.exit(2)，pytest 直接 INTERNALERROR。
if __name__ == "__main__":
    runpy.run_path(str(_TARGET), run_name="__main__")
else:
    # 被 import（测试用）：把 engine 的名字搬进来。
    #
    # ⚠️ 已知局限：拷名字**不共享可变全局**。测试若改 `loop.VERIFY_TIMEOUT_S`
    # 再调函数，改的是本模块的副本，engine 内部读不到。
    # 这类测试应该**直接加载 engine 路径**，而不是经过 shim——
    # 见 test_loop.py 的 LOOP_PY 已改为 engine/loop.py。
    import importlib.util as _ilu

    _spec = _ilu.spec_from_file_location("op_optimize_engine", str(_TARGET))
    _mod = _ilu.module_from_spec(_spec)
    sys.modules["op_optimize_engine"] = _mod
    _spec.loader.exec_module(_mod)
    globals().update(_mod.__dict__)
