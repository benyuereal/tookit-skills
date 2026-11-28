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
import shlex
from functools import lru_cache
from pathlib import Path

_DIR = Path(__file__).resolve().parent / "prompts"
_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")
_CMD_PLACEHOLDER = re.compile(r"\{\{(\w+)(:raw)?\}\}")


@lru_cache(maxsize=1)
def _commands() -> dict:
    import yaml
    return yaml.safe_load((_DIR / "commands.yaml").read_text(encoding="utf-8")) or {}


def command(name: str, **ctx) -> str:
    """渲染命令模板（playbook）。`{{key}}` → shlex 转义；`{{key:raw}}` → 原样。"""
    tpl = _commands().get(name)
    if tpl is None:
        raise KeyError("没有命令模板 %r（可选：%s）"
                       % (name, ", ".join(sorted(_commands()))))
    def _sub(m):
        key, raw = m.group(1), bool(m.group(2))
        if key not in ctx:
            raise KeyError("命令模板 %r 缺参数 %r" % (name, key))
        val = str(ctx[key])
        return val if raw else shlex.quote(val)
    return _CMD_PLACEHOLDER.sub(_sub, tpl)


def render(name: str, **ctx) -> str:
    """读 `prompts/<name>.md`，把 `{{key}}` 替换成 ctx[key]（缺失 → 空串）。"""
    tpl = (_DIR / ("%s.md" % name)).read_text(encoding="utf-8")
    return _PLACEHOLDER.sub(lambda m: str(ctx.get(m.group(1), "")), tpl)
