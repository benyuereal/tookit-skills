#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""floor —— 「还能优化多少」必须出一个数（**入口**）。

判据与算法在 `Plugins/ops/engine/floor.py`，CLI（参数/打印/--write-card）在
`Plugins/ops/engine/cli.py`。本文件只负责把 `python3 -I scripts/floor.py ...`
转发过去，以及保留几个历史名字（`analyze` / `card_spec` / `load_ladder`）给
既有脚本与文档引用。

历史
----
这个文件曾经是**实现**：`analyze()` 在这里，`explore.py` 还要起子进程跑它
再解析 stdout 里的 JSON。那种结构下一次判断要跨两个进程、三份地板实现，
现在都收敛到 engine 了。
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENGINE_DIR = REPO / "Plugins/ops/engine"
# 兼容旧引用：调用方可能读这两个常量去拼路径。
CARDS = REPO / "Plugins/ops/op-atlas/cards"


def _engine(name):
    """按路径加载 engine/<name>.py（见 engine/loop.py::_engine 的说明）。"""
    key = "_ops_engine_" + name
    if key in sys.modules:
        return sys.modules[key]
    import importlib.util
    spec = importlib.util.spec_from_file_location(key, str(ENGINE_DIR / (name + ".py")))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


# 转发历史名字。**它们指向 engine/floor.py 里的同一批对象**（不是副本）。
_floor = _engine("floor")
PATTERN_TO_LAYER = _floor.PATTERN_TO_LAYER
GATE = _floor.GATE
COMPUTE_PEAK = _floor.COMPUTE_PEAK
analyze = _floor.analyze
load_ladder = _floor.load_ladder
load_peak = _floor.load_peak
gate = _floor.gate_of
atlas_spec = _floor.atlas_spec
selftest = _floor.selftest


def card_spec(card_id):
    """atlas 卡 id → 算地板用的 spec。"""
    return _floor.atlas_spec(card_id)


def write_card(card_path, result, spec):
    return _engine("cli").write_card(card_path, result, spec)


def measured_cards():
    return _engine("cli").measured_cards(CARDS)


def main(argv=None):
    return _engine("cli").main_floor(argv)


if __name__ == "__main__":
    sys.exit(main())
