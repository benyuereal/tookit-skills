#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""守 `docs/CODE_MAP.md`：地图不能腐烂。

## 为什么这道测试必须存在

`scripts/` 有 67 个 .py、`engine/` 有 7 个模块。文件一多，"这个文件干什么"
就只能靠文档；而**文档会腐烂** —— 新增文件不更新地图，下次找的人（和 agent）
就会搜到一个不存在的东西，或者根本搜不到。

这道测试把两件事都钉住：

1. **仓库里每个 .py 都在地图里** —— 否则新增模块等于隐身；
2. **地图里每个路径都真实存在** —— 否则地图指向空气。

只守前者会留下死链（删了文件地图没删），只守后者会漏新文件。
两个方向都必须守，这也是 code review 里最容易漏的两件事。

## 地图从哪来

不是手写，而是 `scripts/gen_code_map.py` 生成（分组 + 每行职责都在那个脚本里）。
所以修地图的正确姿势是改生成器再重跑，而不是手改 markdown —— 手改的内容
下次重生成就没了。
"""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MAP = REPO / "docs/CODE_MAP.md"
ENGINE = REPO / "Plugins/ops/engine"
SCRIPTS = REPO / "scripts"

# 地图里认的路径形态：`scripts/xxx.py` 或 `Plugins/ops/engine/xxx.py`
PATH_RE = re.compile(r"`((?:scripts|Plugins/ops/engine)/[A-Za-z0-9_.\-]+\.py)`")


def actual_modules():
    """仓库里真实存在的、应当被地图覆盖的 .py。

    `__init__.py` 不算：它是包标记，没有职责可写。
    """
    got = {"scripts/%s" % p.name for p in SCRIPTS.glob("*.py")}
    got |= {"Plugins/ops/engine/%s" % p.name
            for p in ENGINE.glob("*.py") if p.name != "__init__.py"}
    return got


def listed_modules():
    text = MAP.read_text(encoding="utf-8")
    return set(PATH_RE.findall(text))


def test_code_map_exists():
    assert MAP.is_file(), (
        "缺少 docs/CODE_MAP.md —— 没有地图，%d 个 .py 只能靠文件名猜。"
        "跑 `python3 -I scripts/gen_code_map.py` 生成。"
        % len(actual_modules()))


def test_every_module_is_listed():
    missing = sorted(actual_modules() - listed_modules())
    assert not missing, (
        "这些 .py 没进代码地图（新增模块等于隐身）：\n  %s\n"
        "修法：在 scripts/gen_code_map.py 的 GROUPS 里补一行，再重跑它。"
        % "\n  ".join(missing))


def test_every_listed_path_exists():
    dead = sorted(p for p in listed_modules() if not (REPO / p).is_file())
    assert not dead, (
        "地图里有死链（文件已删但地图还在指）：\n  %s\n"
        "修法：从 scripts/gen_code_map.py 的 GROUPS 里删掉，再重跑它。"
        % "\n  ".join(dead))


def test_engine_layer_has_no_cli_entry_except_loop():
    """分层约定：engine/ 是库，唯一允许自带 CLI 的是 loop.py（它本身是驱动器）。

    这条守的是「实现沉到哪」——`engine/` 里出现 argparse 就说明有人把 CLI
    又写回库里，`scripts/` 的薄壳就没意义了。
    """
    # 允许自带 CLI 的模块（都是有明确理由的可运行工具）：
    #   loop.py    —— 驱动器本身
    #   cli.py     —— explore/floor 的薄 CLI
    #   judge.py   —— 会被拷进 <artifact>/judge/ 当可执行 judge
    #   orchestrator.py —— 迭代循环拥有者（--selfcheck / --demo / --real）
    #   driver.py  —— 兼容 shim（转发到 orchestrator）
    #   iterate.py —— 工作流反转入口
    CLI_OK = {"__init__.py", "loop.py", "cli.py", "judge.py", "driver.py",
              "orchestrator.py", "iterate.py"}
    offenders = []
    for p in sorted(ENGINE.glob("*.py")):
        if p.name in CLI_OK:
            continue
        src = p.read_text(encoding="utf-8", errors="replace")
        if "argparse.ArgumentParser" in src or "import argparse" in src:
            offenders.append(p.name)
    assert not offenders, (
        "这些 engine 模块自己带 CLI：%s —— 分层约定是 engine 只做实现，"
        "CLI 放 scripts/（或 engine/cli.py 统一收）。" % offenders)
