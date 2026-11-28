#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""给 agent 的 prompt：只给结构化反馈与硬约束，不给数字。"""
from __future__ import annotations

import json

from contract import IterationRecord


def render_student_prompt(records: list[IterationRecord], card: str = "",
                          candidate: dict | None = None) -> str:
    """下一轮 prompt：上一轮结构化记录 + 死路清单 + 自己改过什么。"""
    lines = [
        "# 算子实现任务%s" % (("：" + card) if card else ""),
        "",
        "你只负责**写实现**。测量、判定、记账全部由引擎完成；"
        "不要在会话里自己跑 bench 或计时。",
        "",
        "## 落卡硬约束（违反即判 INVALID_NOOP）",
        "- **所有实现必须写进 `<卡>/_common.py`**（`VARIANTS` / `HIP_VARIANTS` + `get_runner` 分支）；"
        "需要新 kernel 就加 `<卡>/kernels/*.hip` 并登记。",
        "- **每完成一段就落卡**，不要等「最后一次性写回」。引擎只认工作区里的改动。",
        "- `/tmp` 只允许临时验证；临时文件请放 `<工作区>/.scratch/`（引擎看得见）。",
        "",
        "## 上一轮（结构化，不是总结）",
    ]
    if not records:
        lines.append("（第一轮，无历史）")
    else:
        r = records[-1]
        lines.append(json.dumps({
            "round": r.iter,
            "verdict": r.verdict,
            "change": r.change,
            "changed_files": r.changed_files,
            "vs_base": r.vs_base,
            "cand_ms": r.cand_ms,
            "base_ms": r.base_ms,
            "utilization": r.utilization,
            "claim": r.claim,
            "acc": r.acc,
        }, ensure_ascii=False, indent=2))
    refuted: list[str] = []
    for r in records:
        refuted.extend(r.refuted_directions or [])
    if candidate:
        lines += ["", "## 本轮要实现的候选",
                  json.dumps({k: candidate.get(k) for k in
                              ("id", "variant", "direction", "expect",
                               "hypothesis", "expect_basis") if candidate.get(k) is not None},
                             ensure_ascii=False, indent=2)]
    lines += ["", "## 已证伪方向（refuted_directions，不要重复试）",
              json.dumps(sorted(set(refuted)), ensure_ascii=False)]
    return "\n".join(lines) + "\n"
