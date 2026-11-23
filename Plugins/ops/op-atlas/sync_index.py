#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sync_index.py —— 从卡片机械生成 INDEX.md 的「算子一览」表。

为什么需要它
------------
INDEX.md 一度是手写的，于是出现了：13 张卡已经 promoted 到 measured，
INDEX 里还写着 draft（`validate.py` 只查「是否登记」，不查「状态是否一致」）。
这就是「一个事实在多处登记、只有一处对」——和 bf16 峰值散落 5 处是同一个病。

现在状态只认卡片：`status` / `ceiling.bound` / `last_verdict.conclusion` 全从
`cards/*.yaml` 读。INDEX 变成生成物，不再有第二个真值源。

validate.py 的 E19 会反向核对：INDEX 表里每行的状态必须等于卡片里的状态。
手改了 INDEX 但没改卡片（或反之），门禁直接红。

用法
----
    python3 -I sync_index.py            # 就地更新 INDEX.md 的表
    python3 -I sync_index.py --check    # 只检查是否同步，不同步退出码 1（CI 用）
"""
import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TABLE_HEAD = "| 算子 | 家族 | 状态 | 首选实现 | 上限 | 上次结论 |"
TABLE_SEP = "|---|---|---|---|---|---|"
CONCL_MAX = 34          # 表格里只放结论前 34 字，全文在卡片里


def _load_yaml(p):
    try:
        import yaml
    except ImportError:                                    # pragma: no cover
        raise SystemExit("需要 PyYAML")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _primary_repo(card):
    """首选实现 = 第一个 tier=primary 的 repo；没有就取第一个 impl 的 repo。"""
    impls = card.get("impls") or []
    for i in impls:
        if i.get("tier") == "primary":
            return i.get("repo") or "—"
    return (impls[0].get("repo") if impls else None) or "—"


def _cell(s, limit=None):
    s = str(s or "").replace("|", "\\|").replace("\n", " ").strip()
    if limit and len(s) > limit:
        s = s[:limit] + "…"
    return s or "—"


def build_rows(cards_dir):
    rows = []
    for p in sorted(cards_dir.glob("*.yaml")):
        c = _load_yaml(p)
        if not c.get("id"):
            continue
        lv = c.get("last_verdict") or {}
        rows.append("| [`%s`](cards/%s.yaml) | %s | %s | %s | %s | %s |" % (
            c["id"], p.stem,
            _cell(c.get("family")),
            _cell(c.get("status")),
            _cell(_primary_repo(c)),
            _cell((c.get("ceiling") or {}).get("bound")),
            _cell(lv.get("conclusion"), CONCL_MAX),
        ))
    return rows


def render(index_text, rows):
    """把 INDEX.md 里从表头到下一个空行/二级标题之间的表替换掉。"""
    lines = index_text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.strip() == TABLE_HEAD), None)
    if start is None:
        raise SystemExit("INDEX.md 里找不到表头：%s" % TABLE_HEAD)
    end = start + 2                      # 跳过表头与分隔行
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    new = lines[:start] + [TABLE_HEAD, TABLE_SEP] + rows + lines[end:]
    return "\n".join(new) + ("\n" if index_text.endswith("\n") else "")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只检查，不同步（CI 用）")
    ap.add_argument("--atlas", default=str(HERE))
    a = ap.parse_args(argv)

    atlas = Path(a.atlas)
    index_p = atlas / "INDEX.md"
    cards_dir = atlas / "cards"
    if not index_p.is_file():
        raise SystemExit("INDEX.md 不存在：%s" % index_p)

    text = index_p.read_text(encoding="utf-8")
    want = render(text, build_rows(cards_dir))

    if want == text:
        print("INDEX.md 与卡片一致（%d 张卡）" % len(list(cards_dir.glob('*.yaml'))))
        return 0
    if a.check:
        old = set(text.splitlines())
        new = set(want.splitlines())
        print("INDEX.md 与卡片不同步。差异行：")
        for line in sorted(old - new)[:10]:
            print("  - %s" % line[:110])
        for line in sorted(new - old)[:10]:
            print("  + %s" % line[:110])
        print("跑 `python3 -I %s` 同步" % Path(__file__).name)
        return 1

    index_p.write_text(want, encoding="utf-8")
    print("已同步 INDEX.md（%d 张卡）" % len(list(cards_dir.glob("*.yaml"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
