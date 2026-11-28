#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""driver 的契约层：一轮的产物、backend 接缝、判定与噪声策略。

**为什么单独一层**：迭代控制流只依赖这里的三个东西（Record / Backend /
decide_verdict）。把它们与 git/agent/GPU 分开，控制流才能用 fake 单测
（`tests/test_driver_fake.py`），也才是 micro-vllm 那种"一个类只干一件事"的形状。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field
from typing import Protocol

# 只认 **代码** 改动：文档（ledger.md/SKILL.md）与配置（variants.yaml/
# candidates.yaml）不算"实现了东西"（BUG B：只改 ledger.md 曾被算成有改动）。
AGENT_CODE_RE = re.compile(
    r"^Plugins/ops/skills/[^/]+/("
    r"_common\.py"
    r"|bench/[^/]+\.py"
    r"|kernels/[^/]+\.(hip|so|sh)"
    r"|tests/[^/]+\.py"
    r")$")

# 引擎自己写的路径：永远不算 agent 改动
ENGINE_PATHS = ("memory/", "results/", ".artifacts/", "Plugins/ops/op-atlas/")
# agent 绝不该改的路径：改了就回滚并判 TAMPERED（它改的是"用来测它的尺子"）
TAMPER_PATHS = ("Plugins/ops/engine/", "scripts/")

NOISE_FLOOR = 0.015          # 1.5%：低于这个的增益不算提升
MAX_PLATEAU = 3              # 连续这么多轮没有 KEEP → plateau
MAX_NOOP = 2                 # 连续这么多轮没有代码改动 → 停


@dataclass
class IterationRecord:
    """一轮的产物（最小字段集）。"""

    iter: int
    round_id: str
    change: str                       # agent 自述「改了什么」，不含数字
    changed_files: list[str] = field(default_factory=list)
    workspace_commit: str = ""
    correctness: bool | None = None
    verdict: str = "NEUTRAL"          # KEEP | REVERT | NEUTRAL | INVALID_NOOP | REFUTED
    base_ms: float | None = None
    cand_ms: float | None = None
    vs_base: float | None = None
    utilization: float | None = None
    floor_us: float | None = None
    acc: dict = field(default_factory=dict)
    claim: str = ""
    refuted_directions: list[str] = field(default_factory=list)
    noise_floor: float = NOISE_FLOOR
    stop_reason: str | None = None    # plateau | impl_fails | rounds_done | noop | agent_timeout
    agent_meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class Backend(Protocol):
    """副作用接缝。控制流只依赖这四个方法，因此可以 fake。"""

    def run_agent(self, prompt: str, role: str = "student") -> str: ...
    def commit(self, message: str) -> str: ...
    def reset_to(self, commit: str) -> None: ...
    def eval(self, tag: str) -> dict: ...


def correctness_of_decision(decision: str | None) -> bool:
    """engine 的 decision → "实现是否仍然正确"。

    **只有**正确性类失败才算 False（REJECTED_ACCURACY / ERROR / IMPL_BROKEN）。
    REJECTED_NO_GAIN / INCONCLUSIVE / CALIBRATION_ONLY 都说明实现是**对的**，
    只是没有收益 —— 按 relaxed-revert 必须保留，交 NEUTRAL，不许 REVERT。
    （实测 2026-11-28：r2 -0.1% 被 REJECTED_NO_GAIN 判成 correctness=False → REVERT，
     把 agent 的改动回滚了，语义反了。）
    """
    d = str(decision or "")
    if not d:
        return False              # 没有记录 = 这轮测量失败
    return d not in ("REJECTED_ACCURACY", "ERROR", "IMPL_BROKEN")


def agent_changes(changed_files: list[str]) -> list[str]:
    """只保留 agent 的**代码**改动（见 AGENT_CODE_RE）。"""
    return [f for f in (changed_files or [])
            if AGENT_CODE_RE.match(str(f))
            and not any(f.startswith(q) for q in ENGINE_PATHS)]


def noise_floor_from(ratios, floor: float = NOISE_FLOOR) -> float:
    """同代 ratio spread（保留给无 --repeat 的旧路径；R1 后主路径用 repeat_spread）。"""
    vals = [float(r) for r in (ratios or []) if isinstance(r, (int, float))]
    if len(vals) < 2:
        return floor
    return max(floor, max(vals) - min(vals))


def decide_verdict(correctness: bool | None, delta: float | None, noise: float,
                   pre_green: bool, claim: str = "", changed_files=None) -> str:
    """relaxed-revert：性能回退保留，只有正确性/证据破坏才 REVERT。"""
    if not agent_changes(list(changed_files or [])):
        return "INVALID_NOOP"
    if correctness is False:
        return "REVERT"
    if claim.startswith(("FAIL:accuracy", "FAIL:evidence")):
        return "REVERT"
    if pre_green or correctness is None:
        return "NEUTRAL"
    if delta is not None and delta < -noise:
        return "KEEP"
    return "NEUTRAL"
