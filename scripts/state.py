#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""状态看板生成器：扫描 op-atlas 卡片的 progress.stage，输出阶段分布。

用法:
    python3 -I scripts/state.py            # 打印看板
    python3 -I scripts/state.py --write    # 写入 docs/STATE.md
    python3 -I scripts/state.py --check    # 与 docs/STATE.md 比对，不一致非 0
    python3 -I scripts/state.py --json     # 机器可读

**唯一权威定义在 Plugins/ops/op-atlas/validate.py 的 STAGES / STAGE_NEEDS /
BLOCKED_BY。** 本文件不复制阶段表，而是 importlib 加载 validate.py 后直接
引用那三个对象。看板里出现的阶段名与前置产出全部由它们派生；validate.py
增减阶段时，这里自动跟随。

分桶规则（刻意把三件事分开，避免「没做」和「做了没记」混在一起）:
  - 有 progress.stage 且合法          → 对应阶段
  - 没有 progress（或没有 stage）      → 未接入
  - progress.stage 不在 STAGES 里      → 非法
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ATLAS = REPO / "Plugins/ops/op-atlas"
DEFAULT_STATE = REPO / "docs/STATE.md"

# 文本骨架放 scripts/templates/state.md；本文件只负责拼动态块。
# `-I` 不把脚本目录放进 sys.path（测试还按路径 importlib 加载），这里补一次。
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.append(str(_SCRIPTS))

from template_render import render as _render_text  # noqa: E402

# 加载 validate.py（它只有常量与函数定义，import 不产生副作用）。
# 这是全仓唯一一份阶段定义；state.py 里没有第二份。
_VALIDATE_PATH = DEFAULT_ATLAS / "validate.py"
_spec = importlib.util.spec_from_file_location("_op_atlas_validate", str(_VALIDATE_PATH))
VALIDATE = importlib.util.module_from_spec(_spec)
sys.modules["_op_atlas_validate"] = VALIDATE
_spec.loader.exec_module(VALIDATE)

# 直接引用（不是拷贝）：`is` 同一对象，validate.py 改了这里立刻改。
STAGES = VALIDATE.STAGES
STAGE_NEEDS = VALIDATE.STAGE_NEEDS
BLOCKED_BY = VALIDATE.BLOCKED_BY

# 仅用于人读的展示标签（不影响分桶）。缺失的 stage 显示为空——
# 阶段集合本身仍然只认 validate.py 的 STAGES。
STAGE_LABELS = {
    "S0": "画像",
    "S1": "天花板",
    "S2": "基线",
    "S3": "归因",
    "S4": "资源画像",
    "S5": "选路",
    "S6": "实验",
    "S7": "收敛",
}


def _load_yaml(path):
    import yaml
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        return None, "%s: %s" % (type(exc).__name__, exc)
    if not isinstance(data, dict):
        return None, "顶层不是 mapping"
    return data, None


def scan(cards_dir):
    """扫描卡片目录，返回分桶结果。

    返回 dict:
      stages      {stage: [card, ...]}          （按 STAGES 全量建键）
      needs       {stage: tuple}                （来自 STAGE_NEEDS）
      blocked     {reason: [card, ...]}         （仅合法 blocked_by）
      unadmitted  [card, ...]                   无 progress 或无 stage
      illegal     [(card, reason), ...]         stage 非法 / progress 非 mapping
      bad_blocked [(card, reason), ...]         blocked_by 不在 BLOCKED_BY
      errors      [(card, reason), ...]         读不出来 / 解析失败
    """
    cards_dir = Path(cards_dir)
    stages = {st: [] for st in STAGES}
    blocked = {}
    unadmitted = []
    illegal = []
    bad_blocked = []
    errors = []

    paths = sorted(cards_dir.glob("*.yaml")) if cards_dir.is_dir() else []
    for path in paths:
        name = path.stem
        card, problem = _load_yaml(path)
        if problem:
            errors.append((name, problem))
            continue
        prog = card.get("progress")
        if prog is None:
            unadmitted.append(name)
            continue
        if not isinstance(prog, dict):
            illegal.append((name, "progress 不是 mapping"))
            continue
        st = prog.get("stage")
        if "stage" not in prog:
            unadmitted.append(name)
            continue
        if st not in STAGES:
            illegal.append((name, "progress.stage=%r 不在 %s" % (st, list(STAGES))))
            continue
        stages[st].append(name)
        bb = prog.get("blocked_by")
        if bb:
            if bb in BLOCKED_BY:
                blocked.setdefault(bb, []).append(name)
            else:
                bad_blocked.append((name, "progress.blocked_by=%r 不在 %s"
                                    % (bb, sorted(BLOCKED_BY))))

    return {
        "cards_dir": str(cards_dir),
        "total": len(paths),
        "stages": stages,
        "needs": {st: tuple(STAGE_NEEDS.get(st, ())) for st in STAGES},
        "blocked": blocked,
        "unadmitted": unadmitted,
        "illegal": illegal,
        "bad_blocked": bad_blocked,
        "errors": errors,
    }


def _needs_text(needs):
    return ", ".join(needs) if needs else "—"


def _stage_line(st, width):
    label = STAGE_LABELS.get(st, "")
    return "%s %-*s" % (st, width, label)


def _bullets(items):
    return "\n".join("- %s" % x for x in items) if items else "- （无）"


def _pair_bullets(pairs):
    return ("\n".join("- %s：%s" % (n, r) for n, r in pairs)
            if pairs else "- （无）")


def _stage_section(st, names, needs):
    title = STAGE_LABELS.get(st)
    lines = ["",
             "### %s%s（%d 张）"
             % (st, (" " + title) if title else "", len(names)),
             "前置产出：%s" % _needs_text(needs)]
    lines.extend("- %s" % name for name in names)
    if not names:
        lines.append("- （无）")
    return "\n".join(lines)


def render(board):
    """把 scan() 的结果渲染成 Markdown（纯函数，--check 依赖其确定性）。"""
    stages = board["stages"]
    needs = board["needs"]
    blocked = board["blocked"]

    width = max([len(STAGE_LABELS.get(st, "")) for st in STAGES] + [0])
    stage_distribution = "\n".join(
        "%s  %3d 张   前置产出：%s"
        % (_stage_line(st, width), len(stages[st]), _needs_text(needs[st]))
        for st in STAGES)

    nb = sum(len(v) for v in blocked.values())
    detail = " / ".join("%s %d" % (r, len(blocked[r])) for r in sorted(blocked))
    blocked_summary = "blocked  %3d 张%s" % (nb,
                                             "（%s）" % detail if detail else "")

    stage_sections = "\n".join(
        _stage_section(st, stages[st], needs[st]) for st in STAGES)

    if blocked:
        blocked_body = "\n".join(
            "\n- %s（%d）：%s" % (reason, len(blocked[reason]),
                                 ", ".join(blocked[reason]))
            for reason in sorted(blocked))
    else:
        blocked_body = "- （无）"

    extra_sections = ""
    if board["bad_blocked"]:
        extra_sections += "\n\n### blocked_by 非法（%d 张）\n%s" % (
            len(board["bad_blocked"]), _pair_bullets(board["bad_blocked"]))
    if board["errors"]:
        extra_sections += "\n\n### 读取失败（%d 张）\n%s" % (
            len(board["errors"]), _pair_bullets(board["errors"]))

    return _render_text(
        "state",
        stage_distribution=stage_distribution,
        blocked_summary=blocked_summary,
        stage_sections=stage_sections,
        blocked_count=nb,
        blocked_body=blocked_body,
        unadmitted_count=len(board["unadmitted"]),
        unadmitted_body=_bullets(board["unadmitted"]),
        illegal_count=len(board["illegal"]),
        illegal_body=_pair_bullets(board["illegal"]),
        extra_sections=extra_sections,
    )


def to_json(board):
    stages = board["stages"]
    return {
        "total": board["total"],
        "stages": [
            {"stage": st,
             "count": len(stages[st]),
             "needs": list(board["needs"][st]),
             "cards": stages[st]}
            for st in STAGES
        ],
        "blocked": {
            "total": sum(len(v) for v in board["blocked"].values()),
            "by_reason": {r: {"count": len(v), "cards": v}
                          for r, v in sorted(board["blocked"].items())},
        },
        "unadmitted": {"count": len(board["unadmitted"]),
                       "cards": board["unadmitted"]},
        "illegal": {"count": len(board["illegal"]),
                    "cards": [{"card": n, "reason": r} for n, r in board["illegal"]]},
        "bad_blocked": {"count": len(board["bad_blocked"]),
                        "cards": [{"card": n, "reason": r}
                                  for n, r in board["bad_blocked"]]},
        "errors": {"count": len(board["errors"]),
                   "cards": [{"card": n, "reason": r} for n, r in board["errors"]]},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="op-atlas 状态看板生成器")
    ap.add_argument("--cards", default=str(DEFAULT_ATLAS / "cards"),
                    help="卡片目录（默认 Plugins/ops/op-atlas/cards）")
    ap.add_argument("--state", default=str(DEFAULT_STATE),
                    help="看板文件路径（默认 docs/STATE.md）")
    ap.add_argument("--write", action="store_true", help="写入看板文件")
    ap.add_argument("--check", action="store_true",
                    help="与看板文件比对；不一致非 0 退出")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    a = ap.parse_args(argv)

    board = scan(a.cards)
    text = render(board)
    state_path = Path(a.state)

    if a.check:
        if not state_path.is_file():
            print("%s 不存在——先运行 --write" % state_path, file=sys.stderr)
            return 2
        current = state_path.read_text(encoding="utf-8")
        if current != text:
            print("%s 已过期（与当前卡片状态不一致）——重新运行 --write"
                  % state_path, file=sys.stderr)
            return 1
        return 0

    if a.write:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(text, encoding="utf-8")
        print("已写入 %s" % state_path, file=sys.stderr)
        return 0

    if a.json:
        print(json.dumps(to_json(board), ensure_ascii=False, indent=2))
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
