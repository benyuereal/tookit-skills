#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""给 agent 的 prompt：**模板在 `instrument/prompts/student.md`**，本文件只填上下文。

这样 prompt 的文本改动（对外契约）不再散落在 if/else 里；缺哪一段，对应的
block 就是空串，模板里自然消失。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from contract import IterationRecord  # noqa: E402
from instrument import render          # noqa: E402

_CAND_KEYS = ("id", "variant", "direction", "expect", "hypothesis", "expect_basis")


def _candidate_block(candidate: dict | None) -> str:
    if not candidate:
        return ("## 本轮目标（candidates.yaml 里没有登记，按下面的最小目标做）\n"
                "- 实现并注册 `--tag` 对应的变体：写进 `<卡>/_common.py` 的 "
                "`VARIANTS`/`HIP_VARIANTS`，并让 `get_runner` 返回可运行实现。\n")
    body = {k: candidate.get(k) for k in _CAND_KEYS if candidate.get(k) is not None}
    return ("## 本轮要实现的候选\n"
            "```json\n%s\n```\n" % json.dumps(body, ensure_ascii=False, indent=2))


def _feedback_block(records: list[IterationRecord]) -> str:
    if not records:
        return "## 上一轮（结构化记录）\n\n（无：这是第 1 轮。）\n"
    r = records[-1]
    payload = {
        "round": r.iter, "verdict": r.verdict, "change": r.change,
        "changed_files": r.changed_files, "vs_base": r.vs_base,
        "cand_ms": r.cand_ms, "base_ms": r.base_ms,
        "utilization": r.utilization, "claim": r.claim, "acc": r.acc,
    }
    return ("## 上一轮（结构化记录）\n"
            "```json\n%s\n```\n" % json.dumps(payload, ensure_ascii=False, indent=2))


def _refuted_block(records: list[IterationRecord]) -> str:
    got = sorted({d for r in records for d in (r.refuted_directions or [])})
    return ("## 已证伪方向（refuted_directions，不要重复试）\n"
            + ("\n".join("- `%s`" % x for x in got) if got else "（无）") + "\n")


def render_student_prompt(records: list[IterationRecord], card: str = "",
                          candidate: dict | None = None,
                          extra_task_lines: list[str] | None = None,
                          round_instruction: str | None = None) -> str:
    """下一轮 prompt。段落由 block 生成；文本在 student.md 模板里。"""
    task_lines = ["- 卡目录：`%s`" % card] if card else []
    task_lines += [str(x) for x in (extra_task_lines or [])]
    tag = (candidate or {}).get("id") or "<见 --tag>"
    return render(
        "student",
        card_suffix=("：" + Path(card).name) if card else "",
        task_lines="\n".join(task_lines),
        candidate_block=_candidate_block(candidate),
        feedback_block=_feedback_block(records),
        refuted_block=_refuted_block(records),
        round_block=(round_instruction
                     or "## 本轮\n实现候选 `%s` 的一个改动，然后退出。"
                        "标准输出写「改了什么」。" % tag) + "\n",
    )
