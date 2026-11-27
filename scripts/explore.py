#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""explore —— 没有候选时，从**通用信号**出题；不拒绝，也不放松六道门（**入口**）。

## 为什么需要它（2026-11-27 实测）

C 臂在三臂测试里**唯一一次**执行 loop.py 是：

    loop.py -C Plugins/ops/skills/op-moe-topk-gating status
    → 候选  待做 0 / 已判死 0 / 停滞可作父代 0 / 共记录 0 条

然后它再也没碰过 loop.py，转身自己写 bench。全仓 16 个待做候选里，能映射到
**已实现变体**的只有 4 个。也就是说：引擎本身没坏，是"没活干"被 agent 正确地
读成了死胡同——这是引擎的失败，不是 agent 的。

## 现在这个文件只剩入口

    engine/probe.py    采信号（derive 模型 / runs 达成 / 变体清单 / 方向预算）
    engine/propose.py  信号 → 候选（规则与话术）+ 人读渲染
    engine/floor.py    地板（通用地板与 derive 模型都从它取，唯一实现）
    engine/card.py     读卡（candidates.yaml 两种格式的兼容）

它们原来都在这个文件里（597 行）。拆开的理由见各模块头部说明；
本文件保留名字与 CLI 表面，行为（含人读文本）一个字没变。

## 用法

    python3 -I scripts/explore.py --card Plugins/ops/skills/op-flash-mla            # 只看
    python3 -I scripts/explore.py --card Plugins/ops/skills/op-flash-mla --apply    # 写 overlay
    python3 -I scripts/explore.py --card ... --shape decode-m32 --json

`--apply` 写 `runs/advance/overlay.yaml`，`loop.py` 的 `load_candidates` 会合并它
并把这些候选标成 `_origin=advance-overlay`。**跑不跑仍由 loop.py 的六道门决定。**
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = REPO_ROOT / "Plugins/ops/engine"


def _engine(name):
    """按路径加载 engine/<name>.py（见 engine/loop.py::_engine 的说明）。

    本脚本用 `python3 -I` 跑，`Plugins/ops` 又不是包，所以既不能相对导入
    也不能裸 import —— 与仓库里既有的几处按路径加载是同一个原因。
    """
    key = "_ops_engine_" + name
    if key in sys.modules:
        return sys.modules[key]
    import importlib.util
    spec = importlib.util.spec_from_file_location(key, str(ENGINE_DIR / (name + ".py")))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


# 历史名字：既有脚本/测试点过这些名字，转发到新位置（同一批对象，不是副本）。
_probe = _engine("probe")
_propose = _engine("propose")
load_surfaces = _probe.load_surfaces
achieved_us = _probe.achieved_us
direction_budget = _probe.direction_budget
available_variants = _probe.available_variants
collect_signals = _probe.collect
propose = _propose.propose
render = _propose.render


def explore(card, shape=None, limit=5):
    return _engine("cli").explore_report(card, shape, limit)


def apply_overlay(card, props):
    """写进 runs/advance/overlay.yaml（与 advance 的 forge 共用同一个实现）。"""
    return _engine("ledger").write_overlay(card, props)


def main(argv=None):
    return _engine("cli").main_explore(argv)


if __name__ == "__main__":
    raise SystemExit(main())
