#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""evals 静态检查器：校验各 skill 的 evals/evals.json 结构与全库 id 唯一性。

不跑模型、不做语义评测——只做静态质量门（供 CI 与发布前自检）：
  1. 每个 evals/evals.json 可解析（容忍 UTF-8 BOM）、skill 字段与所在目录名一致；
  2. cases 非空，且 true 案例数 >= 3、false 案例数 >= 2（对齐 4.11.0 起的既有约定）；
  3. id 形如 <skill>-t1/-f1 且编号从 1 连续；true 案例必须带非空 expect（可选 forbid：
     路由对之后判回答里的禁用说法，`re:` 前缀按正则校验），false 案例不得带 expect/forbid；
  4. id 全库唯一（防止跨文件撞车导致评测报告无法定位）；
  5. description 合规（docs/STANDARDS.md：触发条件 + 不适用反例）。有 evals 的 skill
     必须合规（这些描述正是被动态评测判分的那段话）；没有 evals 的只统计不阻塞，
     `--strict-description` 可把全库都变成阻塞（用于逐步收敛）。

用法：
    python scripts/run_evals.py [--root <包根>] [--show N] [--strict-description]
PASS 退 0、FAIL 退 1。全库扫描，无需参数即可在仓库根运行。
"""
import argparse
import io
import json
import os
import re
import sys

SKIP_DIR_NAMES = (".git", "__pycache__", "node_modules", ".venv", "vendor_dcuprofiletools")
ID_RE = re.compile(r"^(?P<skill>.+)-(?P<kind>[tf])(?P<num>\d+)$")

# description 的两个必备面（docs/STANDARDS.md：「触发条件 + 不适用反例」）。
# 标的是**语义标记**，不是措辞：agent 靠这两句决定「该不该拉起本 skill」。
TRIGGER_MARKERS = ("触发", "何时用", "适用场景", "加载时机", "when to use", "use when")
BOUNDARY_MARKERS = ("不适用", "不要用", "别用", "不属于", "not for", "don't use")


def read_description(skill_md):
    """取 SKILL.md frontmatter 的 description；没有 frontmatter/字段返回 None。"""
    try:
        with io.open(skill_md, "r", encoding="utf-8-sig") as f:
            text = f.read()
    except OSError:
        return None
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    if end < 0:
        return None
    m = re.search(r"(?m)^description:\s*(.*)$", text[3:end])
    if not m:
        return None
    value = m.group(1).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1]
    return value or None


def description_gaps(desc):
    """返回缺的语义面（"触发条件" / "不适用反例"）；空列表 = 合规。"""
    text = (desc or "").lower()
    gaps = []
    if not any(k in text for k in TRIGGER_MARKERS):
        gaps.append("触发条件")
    if not any(k in text for k in BOUNDARY_MARKERS):
        gaps.append("不适用反例")
    return gaps


def iter_skill_dirs(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
        # 只认 <...>/skills/<skill>/SKILL.md：包根与插件根的 SKILL.md 是入口文档，不是 skill
        if "SKILL.md" in filenames and os.path.basename(os.path.dirname(dirpath)) == "skills":
            yield dirpath


def check_one(path, skill_dir_name):
    """返回 (problems, n_true, n_false)。problems 为 (标签, 消息) 列表。"""
    problems = []
    try:
        with io.open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except (ValueError, OSError) as e:
        return [("parse", "evals.json 解析失败: %s" % e)], 0, 0
    if not isinstance(data, dict):
        return [("schema", "顶层必须是 JSON object")], 0, 0

    if data.get("skill") != skill_dir_name:
        problems.append(("schema", "skill 字段 %r 与目录名 %r 不一致" % (data.get("skill"), skill_dir_name)))
    if not str(data.get("description") or "").strip():
        problems.append(("schema", "缺 description（说明 true/false 语义）"))

    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        return problems + [("schema", "cases 必须是非空数组")], 0, 0

    n_true = n_false = 0
    seen_num = {"t": set(), "f": set()}
    for i, case in enumerate(cases):
        where = "cases[%d]" % i
        if not isinstance(case, dict):
            problems.append((where, "案例必须是 object"))
            continue
        cid = case.get("id")
        m = ID_RE.match(str(cid or ""))
        if not m or m.group("skill") != skill_dir_name:
            problems.append((where, "id %r 不符合 <skill>-tN/-fN 且前缀=skill 名" % cid))
            continue
        kind, num = m.group("kind"), int(m.group("num"))
        seen_num[kind].add(num)
        if str(case.get("prompt") or "").strip() == "":
            problems.append((where, "案例 %s 缺 prompt" % cid))
        if kind == "t":
            n_true += 1
            if not str(case.get("expect") or "").strip():
                problems.append((where, "true 案例 %s 缺非空 expect（应写明命中要点）" % cid))
            forbid = case.get("forbid")
            if forbid is not None:
                if not isinstance(forbid, list) or not all(
                        isinstance(x, str) and x.strip() for x in forbid):
                    problems.append((where, "true 案例 %s 的 forbid 必须是非空字符串数组" % cid))
                else:
                    for x in forbid:
                        if x.startswith("re:"):
                            try:
                                re.compile(x[3:])
                            except re.error as e:
                                problems.append((where, "true 案例 %s 的 forbid 正则不合法：%s"
                                                 % (cid, e)))
                        elif len(x.strip()) < 4:
                            problems.append((where, "true 案例 %s 的 forbid 条目太短（≤3 字）："
                                                    "写「错话」本身而不是关键词，否则会误伤正确回答"
                                             % cid))
        else:
            n_false += 1
            if "expect" in case:
                problems.append((where, "false 案例 %s 不应带 expect（只给 id+prompt）" % cid))
            if "forbid" in case:
                problems.append((where, "false 案例 %s 不应带 forbid（没加载 skill，不判内容）" % cid))

    for kind, minimum, label in (("t", 3, "true"), ("f", 2, "false")):
        nums = sorted(seen_num[kind])
        if len(nums) < minimum:
            problems.append(("coverage", "%s 案例仅 %d 个（约定至少 %d）" % (label, len(nums), minimum)))
        elif nums != list(range(1, len(nums) + 1)):
            problems.append(("coverage", "%s 案例编号不连续（应为 1..%d，实为 %s）" % (label, len(nums), nums)))
    return problems, n_true, n_false


def main():
    ap = argparse.ArgumentParser(description="dcu-toolkit-skills evals 静态检查器")
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--show", type=int, default=50, help="每类最多展示条数")
    ap.add_argument("--strict-description", action="store_true",
                    help="没有 evals 的 skill 描述不合规也算 FAIL（默认只统计）")
    args = ap.parse_args()
    root = os.path.abspath(args.root)

    skill_dirs = sorted(iter_skill_dirs(root))
    files, problems, dup_ids = [], [], {}
    n_true = n_false = 0
    n_desc_ok = n_desc_report = 0
    for dirpath in skill_dirs:
        p = os.path.join(dirpath, "evals", "evals.json")
        has_evals = os.path.isfile(p)
        if has_evals:
            rel = os.path.relpath(p, root).replace(os.sep, "/")
            files.append(rel)
            ps, t, fnum = check_one(p, os.path.basename(dirpath))
            n_true += t
            n_false += fnum
            problems.extend((rel, tag, msg) for tag, msg in ps)
            with io.open(p, "r", encoding="utf-8-sig") as fh:
                for case in (json.load(fh).get("cases") or []):
                    cid = case.get("id")
                    dup_ids.setdefault(cid, []).append(rel)

        # description 合规：有 evals 的一律阻塞；其余只统计（--strict-description 时阻塞）
        skill = os.path.basename(dirpath)
        desc = read_description(os.path.join(dirpath, "SKILL.md"))
        gaps = description_gaps(desc)
        if not gaps:
            n_desc_ok += 1
        elif has_evals or args.strict_description:
            problems.append((skill, "description",
                             "缺 %s（docs/STANDARDS.md：触发条件 + 不适用反例）" % " / ".join(gaps)))
        else:
            n_desc_report += 1

    dups = {k: v for k, v in dup_ids.items() if len(v) > 1}
    for cid, locs in sorted(dups.items()):
        problems.append(("-", "dup-id", "id %r 在 %d 个文件重复: %s" % (cid, len(locs), ", ".join(locs))))

    print("[evals] 覆盖 skill 数: %d（true 案例合计 %d / false 合计 %d）" % (len(files), n_true, n_false))
    print("[description] 全库 %d 个 skill：合规 %d，缺语义面 %d（%s）"
          % (len(skill_dirs), n_desc_ok, n_desc_report,
             "已全部阻塞" if args.strict_description else "无 evals 的只统计，不阻塞"))
    for rel, tag, msg in problems[:args.show]:
        print("  %s: %s: %s" % (rel, tag, msg))
    if len(problems) > args.show:
        print("  ... 其余 %d 条用 --show 调大" % (len(problems) - args.show))
    print("EVALS: %s" % ("FAIL" if problems else "PASS"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
