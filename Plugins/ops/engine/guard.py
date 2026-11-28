#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""命令级守卫：测量只能走 `loop.py run`，不许直接跑卡片的 bench。

来源：`scripts/advance.py::_guard_no_bench`（已在两轮实证里拦住过绕行）。
提取成公共模块，是为了让 driver / advance 共用同一套判据，而不是各写一份。

**只读放行**：`ls bench/`、`cat bench/ab.py` 这类查看不算执行 ——
原实现一律拒绝，会把"看看有哪些文件"也挡掉（实测误伤）。
"""
from __future__ import annotations

import re

# 只读命令前缀：查看 bench 目录/源码是允许的
_READONLY = re.compile(
    r"^\s*(cat|sed|head|tail|less|more|grep|rg|ls|find|wc|stat|file|du|tree)\b")
# 真正"执行"的特征：跑 ab.py / acc_probe.py
_EXEC_TARGETS = ("ab.py", "acc_probe.py")


class GuardViolation(RuntimeError):
    """命令试图绕过 loop.py 直接测量。"""


def guard_no_bench(cmd) -> None:
    """命中「直接跑 bench」时抛 GuardViolation；只读命令放行。

    `cmd` 可以是 argv 列表或整条 shell 字符串（与 advance 的调用方式一致）。
    """
    s = " ".join(cmd) if isinstance(cmd, (list, tuple)) else str(cmd)
    low = s.lower()
    if _READONLY.match(low):
        return
    # /bench/ 且带执行语义（python/bash/sh/hipcc），或直接点名 ab.py / acc_probe.py
    hit_target = any(t in low for t in _EXEC_TARGETS)
    hit_exec = re.search(r"\b(python[0-9.]*|bash|sh|hipcc|pytest|torchrun)\b", low) \
        and "/bench/" in low
    if hit_target or hit_exec:
        raise GuardViolation(
            "拒绝直接跑 bench（%s）：测量只能经 `loop.py run`，"
            "否则六道门与账本都被绕过" % s)
