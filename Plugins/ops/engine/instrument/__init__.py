#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""prompt 模板渲染器：模板放 `prompts/*.md`，代码只负责填上下文。

为什么单独一层（业主 2026-11-28）：
  · prompt 是**对外文本**，改一句话就是改契约；放在代码里被 if/else 包着，
    既难读也难对比。
  · 模板用 `{{name}}` 占位，缺的键渲染成空串 —— 没有一堆条件分支。

用法：
    from instrument import render
    text = render("student", card=..., candidate_block=..., ...)
"""
from __future__ import annotations

import re
from pathlib import Path

_DIR = Path(__file__).resolve().parent / "prompts"
_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


def render(name: str, **ctx) -> str:
    """读 `prompts/<name>.md`，把 `{{key}}` 替换成 ctx[key]（缺失 → 空串）。"""
    tpl = (_DIR / ("%s.md" % name)).read_text(encoding="utf-8")
    return _PLACEHOLDER.sub(lambda m: str(ctx.get(m.group(1), "")), tpl)
