#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""scripts 侧的模板渲染器（与 engine/instrument 同语义）。

文本放 `scripts/templates/<name>.md`，`{{key}}` 占位，缺键 → 空串。
与 engine 侧共用同一套约定，避免每个脚本各写一份 append 墙。
"""
from __future__ import annotations

import re
from pathlib import Path

_DIR = Path(__file__).resolve().parent / "templates"
_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


def render(name: str, **ctx) -> str:
    tpl = (_DIR / ("%s.md" % name)).read_text(encoding="utf-8")
    return _PLACEHOLDER.sub(lambda m: str(ctx.get(m.group(1), "")), tpl)
