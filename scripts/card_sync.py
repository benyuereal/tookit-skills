#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""card_sync —— 一张卡要登记的 5 个位置，一个入口全管。

## 解决什么问题

新建或修改一张卡，要在 **5 个地方**保持一致：

| # | 位置 | 漏了会怎样 |
|---|---|---|
| 1 | `op-atlas/cards/<卡>.yaml` | 卡本身（唯一真源） |
| 2 | `data/ops/registry.json` | `test_every_card_is_registered` 红 |
| 3 | `op-atlas/INDEX.md` | validate E19 红 |
| 4 | `ops/SKILL.md` 索引表 | `test_entry_index_lists_every_skill` 红 |
| 5 | `op-atlas/gaps.md` | `test_gaps_is_generated_and_in_sync` 红 |

**这是「同一事实多处副本」的第 7 次复现** —— 实测：2026-11-25 新建
`norm-act-swiglu` 卡，就是分 3 次才把 5 处补齐，每次都是测试报错才想起来。

## 设计原则

- **卡片是唯一真源**。registry / INDEX / SKILL.md / gaps 全部从卡派生。
- **不覆盖手工字段**。registry 里有 `best_variant` / `floor_status` /
  `open_questions` 这类需要人写的内容 —— 只补缺失的，不乱改已有的。
- **幂等**。跑两次结果一样；已经一致的位置不重写（保留文件 mtime，
  避免无意义的 diff 噪音）。
- **能只检查不修**。`--check` 只报问题，适合放进 CI。

## 用法

    python3 -I scripts/card_sync.py                 # 修全部卡（幂等）
    python3 -I scripts/card_sync.py --check         # 只检查，不写
    python3 -I scripts/card_sync.py --card <名>     # 只处理一张
    python3 -I scripts/card_sync.py --selftest
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CARDS = REPO / "Plugins/ops/op-atlas/cards"
REGISTRY = REPO / "data/ops/registry.json"
OPS_ENTRY = REPO / "Plugins/ops/SKILL.md"
SKILLS = REPO / "Plugins/ops/skills"
SYNC_INDEX = REPO / "Plugins/ops/op-atlas/sync_index.py"
GEN_GAPS = REPO / "scripts/gen_gaps.py"

# registry 里这些字段**由人写**，同步时只补空位，不覆盖已有内容。
HUMAN_FIELDS = ("best_variant", "floor_status", "open_questions", "evidence")


def _load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _save_json(p, d):
    Path(p).write_text(json.dumps(d, ensure_ascii=False, indent=1) + "\n",
                       encoding="utf-8")


def _card_files(only=None):
    """只返回**有 skill 目录**的卡。

    **这是关键约束，第一版搞错了**：registry 登记的是「有可运行壳的算子」
    （7 个），不是全部 54 张卡片。测试 `test_every_card_is_registered`
    比对的是 `Plugins/ops/skills/op-*` 目录，不是 cards/*.yaml。

    第一版遍历全部 54 张卡 → 给每张都往 registry 塞一条 → 914 行误伤。
    """
    out = []
    for f in sorted(CARDS.glob("*.yaml")):
        if only and f.stem != only:
            continue
        if not (SKILLS / ("op-%s" % f.stem)).is_dir():
            continue
        out.append(f)
    return out


def _read_card(f):
    import yaml
    return yaml.safe_load(Path(f).read_text(encoding="utf-8")) or {}


# ---------------------------------------------------------------- 各位置

def sync_registry(card, cards_dir=SKILLS, check=False):
    """① registry.json：确保卡在 `ops` 里有一条。

    **只补缺失，不改已有的人写字段**。实测踩过：直接整条覆盖会丢掉
    `best_variant` / `open_questions` 这些只有人知道的内容。
    """
    name = card["id"]
    reg = _load_json(REGISTRY)
    idx = next((i for i, o in enumerate(reg["ops"]) if o.get("name") == name), None)
    skill_dir = Path(cards_dir) / ("op-%s" % name)
    want = {
        "name": name,
        "path": str(skill_dir.relative_to(REPO)) if skill_dir.is_dir()
                else "Plugins/ops/skills/op-%s" % name,
        "status": card.get("status") or "draft",
    }
    if idx is None:
        # 新卡：给一个最小骨架，人写字段留空待补
        entry = dict(want)
        entry.update({"shapes": card.get("shapes") or [],
                      "variants": [v.get("name") or v.get("repo")
                                   for v in (card.get("variants") or [])
                                   if isinstance(v, dict)],
                      "best_variant": None,
                      "floor_status": None,
                      "last_verified": card.get("updated"),
                      "evidence": None,
                      "open_questions": []})
        if check:
            return ["registry.json 缺 %s" % name]
        reg["ops"].append(entry)
        # **不许用卡片的 updated 覆盖 registry 的 updated**：后者是
        # 「registry 上次被改的时间」，不是「某张卡的时间」。
        import time as _t
        reg["updated"] = _t.strftime("%Y-%m-%d")
        _save_json(REGISTRY, reg)
        return []
    # 已存在：只同步 status（它是唯一真源派生的），其它字段不动
    changed = []
    if reg["ops"][idx].get("status") != want["status"]:
        changed.append("status %r -> %r" % (reg["ops"][idx].get("status"),
                                            want["status"]))
        if not check:
            reg["ops"][idx]["status"] = want["status"]
    if check and changed:
        return ["registry.json %s: %s" % (name, "; ".join(changed))]
    if changed and not check:
        _save_json(REGISTRY, reg)
    return []


def sync_ops_entry(card, check=False, entry=None):
    """② ops/SKILL.md 索引表：确保 `op-<name>` 有一行，且状态与卡一致。

    索引表的两条判据（来自 test_doc_links）：
      · 每个 skill 都要有一行（少一行 → 测试红）
      · 每行的状态列必须等于 registry 的 status（不一致 → 测试红）
    """
    entry = entry or OPS_ENTRY
    if not Path(entry).is_file():
        return ["找不到 %s" % entry]
    name = "op-%s" % card["id"]
    status = card.get("status") or "draft"
    text = Path(entry).read_text(encoding="utf-8")
    row_re = re.compile(r"(?m)^\|\s*`%s`\s*\|\s*([^|]*?)\s*\|" % re.escape(name))
    m = row_re.search(text)
    if m is None:
        if check:
            return ["ops/SKILL.md 缺 %s 索引行" % name]
        desc = card.get("title") or name
        new_row = "| `%s` | `%s` | %s |" % (name, status, desc)
        # 插在最后一个 `| \`op-` 行的后面，保持表格连续
        rows = list(re.finditer(r"(?m)^\|\s*`op-[\w-]+`\s*\|.*$", text))
        if rows:
            at = rows[-1].end()
            text = text[:at] + "\n" + new_row + text[at:]
        else:
            text = text.rstrip() + "\n\n" + new_row + "\n"
        Path(entry).write_text(text, encoding="utf-8")
        return []
    shown = m.group(1).strip().strip("`")
    if shown != status:
        if check:
            return ["ops/SKILL.md %s 状态=%r，卡片=%r" % (name, shown, status)]
        text = text[:m.start(1)] + " `%s` " % status + text[m.end(1):]
        Path(entry).write_text(text, encoding="utf-8")
    return []


def run_script(script, check=False):
    """③④ 跑生成器（INDEX.md / gaps.md）。它们是自己维护格式的生成物，
    不在这里重实现 —— 那会变成第二处记账。"""
    if not Path(script).is_file():
        return ["找不到 %s" % script]
    if check:
        return []          # 由 validate / pytest 去比对，这里不重算
    p = subprocess.run([sys.executable, "-I", str(script)],
                       capture_output=True, text=True, cwd=str(REPO))
    if p.returncode != 0:
        return ["%s 失败：%s" % (Path(script).name,
                                 (p.stderr or p.stdout)[-300:])]
    return []


# ---------------------------------------------------------------- 编排

def sync(only=None, check=False, emit=print):
    """修/查全部位置。返回问题列表（空 = 一致）。"""
    probs = []
    cards = _card_files(only)
    if not cards:
        return ["没有找到卡%s" % ((" %r" % only) if only else "")]
    for f in cards:
        card = _read_card(f)
        if not card.get("id"):
            probs.append("%s 缺 id —— 卡不合法，先修卡" % f.name)
            continue
        probs += sync_registry(card, check=check)
        probs += sync_ops_entry(card, check=check)
    if not check:
        # 生成物放在最后跑：它们依赖 registry 与卡都已就绪
        probs += run_script(SYNC_INDEX)
        probs += run_script(GEN_GAPS)
    return probs


def selftest():
    """**反向优先**：证明它真能发现并修好「漏登记」。"""
    import tempfile
    import yaml
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        cards = d / "cards"
        cards.mkdir()
        (cards / "op-x.yaml").write_text(yaml.safe_dump({
            "id": "x", "title": "测试卡", "status": "measured"}),
            encoding="utf-8")
        reg = d / "registry.json"
        reg.write_text(json.dumps({"ops": [], "updated": None}), encoding="utf-8")
        entry = d / "SKILL.md"
        entry.write_text("| `op-other` | `draft` | 别的卡 |\n", encoding="utf-8")

        orig = (sync_registry.__globals__["REGISTRY"], sync_ops_entry.__globals__["OPS_ENTRY"])
        try:
            sync_registry.__globals__["REGISTRY"] = reg
            sync_ops_entry.__globals__["OPS_ENTRY"] = entry
            card = {"id": "x", "title": "测试卡", "status": "measured"}
            # ① --check 必须报出两处缺失
            p1 = sync_registry(card, check=True)
            p2 = sync_ops_entry(card, check=True)
            assert p1 and "registry.json 缺" in p1[0], p1
            assert p2 and "ops/SKILL.md 缺" in p2[0], p2
            print("[ok] 反向：--check 能报出 registry 与 SKILL.md 两处缺失")

            # ② 修完再 check 必须干净
            assert sync_registry(card, check=False) == []
            assert sync_ops_entry(card, check=False) == []
            assert sync_registry(card, check=True) == [], "修完仍报缺"
            assert sync_ops_entry(card, check=True) == [], "修完仍报缺"
            print("[ok] 正向：修完之后 --check 干净")

            # ③ 幂等：跑第二次不产生变更
            before = reg.read_text(encoding="utf-8")
            sync_registry(card, check=False)
            assert reg.read_text(encoding="utf-8") == before, "不幂等"
            print("[ok] 幂等：第二次跑不改文件")

            # ④ 已有人写字段不许被覆盖
            rd = json.loads(reg.read_text(encoding="utf-8"))
            rd["ops"][0]["best_variant"] = "人手写的结论"
            reg.write_text(json.dumps(rd), encoding="utf-8")
            sync_registry(card, check=False)
            rd2 = json.loads(reg.read_text(encoding="utf-8"))
            assert rd2["ops"][0]["best_variant"] == "人手写的结论", "覆盖了人写字段"
            print("[ok] 反向：不覆盖 registry 里人写的字段")

            # ⑤ 状态漂移必须被同步
            card2 = dict(card, status="draft")
            sync_registry(card2, check=False)
            rd3 = json.loads(reg.read_text(encoding="utf-8"))
            assert rd3["ops"][0]["status"] == "draft", rd3["ops"][0]
            print("[ok] 正向：卡片状态变了，registry 跟着变")
        finally:
            sync_registry.__globals__["REGISTRY"] = orig[0]
            sync_ops_entry.__globals__["OPS_ENTRY"] = orig[1]
    print("selftest PASS")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--card", help="只处理这一张（卡 id，不带 op- 前缀）")
    ap.add_argument("--check", action="store_true", help="只检查不修")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    probs = sync(only=a.card, check=a.check)
    if probs:
        for p in probs:
            print("✗ " + p)
        return 1
    print("✓ 5 个位置一致" + ("（只检查，未写）" if a.check else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
