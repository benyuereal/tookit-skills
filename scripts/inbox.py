#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inbox：把一次 agent 产出追加成一条**待审记录**（append-only）。

设计依据：`/root/atlas-handoff/WORKFLOW-UPGRADE-PLAN.md` §3（事后审计）。

为什么是追加而不是审批流：
  一张卡串行跑，每轮 patch 都等人批，eval_rate 会被打到接近零。
  Agent 先落 Artifact、先过机器判据（E39–E43 / pytest / floor 复算），
  再把「一次产出」压成一行 JSON 追加到 docs/INBOX.jsonl，人**事后**批量审。

两条写死的规则：
  ① INBOX.jsonl **只追加**，脚本永远不打开旧行来改；
  ② `needs_human` 由 gates 计算，**存在任何一个 fail 就强制 yes**——
     人不许把它写成 no（也没有这个参数）。

用法：
  python3 -I scripts/inbox.py append --card rmsnorm-fused --stage S6 \\
      --claim "融合残差方案 remaining 19.8%" \\
      --gates E41=pass,E42=pass,E43=pass,pytest=pass,floor_recompute=pass \\
      --artifacts cards/rmsnorm-fused.yaml#last_verdict.resources

  python3 -I scripts/inbox.py list [--needs-human] [--json]
"""
import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_INBOX = REPO / "docs/INBOX.jsonl"

# 与 validate.py 的 STAGES 一致（只读引用，不改它）
STAGES = ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7")

# gate 名：E 编号 + 三个固定脚本判据。写错名字的 gate 等于没跑。
GATE_NAME_RE = re.compile(r"^E\d+$")
EXTRA_GATES = {"pytest", "floor_recompute", "validate", "check_semantics"}
GATE_VALUES = {"pass", "fail"}


class DuplicateId(ValueError):
    """同一个 id 已经存在——拒绝重复追加（一次产出只审一次）。"""


def _fail(msg):
    raise ValueError(msg)


def _clean_str(value, what):
    if not isinstance(value, str) or not value.strip():
        _fail("%s 必须是非空字符串" % what)
    if "\n" in value or "\r" in value:
        _fail("%s 不许含换行（一行一条记录）" % what)
    return value.strip()


def parse_gates(items):
    """把 ["E41=pass,E42=fail", ...] 解析成 {"E41": "pass", ...}。

    拒绝：空、缺 '='、未知 gate 名、非法取值。gate 名写错不等于通过。
    """
    gates = {}
    if not items:
        _fail("--gates 不能为空：没有任何机器判据的产出不许进 Inbox")
    for raw in items:
        for part in str(raw).split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                _fail("gate %r 缺 '='（格式 NAME=pass|fail）" % part)
            name, value = part.split("=", 1)
            name, value = name.strip(), value.strip().lower()
            if not (GATE_NAME_RE.match(name) or name in EXTRA_GATES):
                _fail("未知 gate 名 %r（只接受 E<数字> 或 %s）"
                      % (name, sorted(EXTRA_GATES)))
            if value not in GATE_VALUES:
                _fail("gate %s 的值 %r 非法（只允许 pass/fail）" % (name, value))
            if name in gates:
                _fail("gate %s 重复出现" % name)
            gates[name] = value
    if not gates:
        _fail("--gates 解析后为空")
    return gates


def parse_artifacts(items):
    """artifact 可逗号分隔，也可重复传；空列表允许（例如纯状态记录）。"""
    out = []
    for raw in items or []:
        for part in str(raw).split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out


def build_entry(card, stage, claim, gates, artifacts=None, ts=None):
    """构造一行记录。needs_human **只由 gates 推出来**，不接受外部指定。"""
    card = _clean_str(card, "card")
    if "/" in card:
        _fail("card 不许含 '/'（id 用 '/' 分段）")
    if stage not in STAGES:
        _fail("stage=%r 非法（只允许 %s）" % (stage, list(STAGES)))
    claim = _clean_str(claim, "claim")
    if not isinstance(gates, dict) or not gates:
        _fail("gates 必须是非空 mapping")
    for name, value in gates.items():
        if value not in GATE_VALUES:
            _fail("gate %s 的值 %r 非法" % (name, value))

    if ts is None:
        ts = datetime.now().replace(microsecond=0).isoformat()
    else:
        ts = _clean_str(ts, "ts")
        if "/" in ts:
            _fail("ts 不许含 '/'")

    failed = sorted(name for name, value in gates.items() if value == "fail")
    return {
        "id": "%s/%s/%s" % (ts, card, stage),
        "card": card,
        "stage": stage,
        "artifacts": parse_artifacts(artifacts),
        "gates": dict(gates),
        "claim": claim,
        # 硬规则：有 fail 就强制 yes；全 pass 才是 no。
        "needs_human": "yes" if failed else "no",
    }


def read_entries(path):
    """读 INBOX。文件不存在 = 空（首次运行不是错误）。

    坏行不沉默跳过：一条解析不了的行会让后续「同 id 去重」失效，
    宁可报错，也不要在坏数据上继续追加。
    """
    path = Path(path)
    if not path.exists():
        return []
    entries = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            _fail("%s 第 %d 行不是合法 JSON：%s" % (path, lineno, exc))
        if not isinstance(obj, dict) or "id" not in obj:
            _fail("%s 第 %d 行不是带 id 的记录" % (path, lineno))
        entries.append(obj)
    return entries


def append_entry(path, entry):
    """追加一行。同 id 已存在 → DuplicateId；旧字节一个也不动。"""
    path = Path(path)
    existing = read_entries(path)
    ids = {e.get("id") for e in existing}
    if entry["id"] in ids:
        raise DuplicateId("id %r 已存在，拒绝重复追加" % entry["id"])

    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = ""
    if path.exists():
        text = path.read_text(encoding="utf-8")
        # 只可能补一个换行分隔符；已有内容逐字节保留。
        if text and not text.endswith("\n"):
            prefix = "\n"
    line = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
    with path.open("a", encoding="utf-8") as fh:
        fh.write(prefix + line + "\n")
    return entry


def list_entries(path, needs_human=False):
    entries = read_entries(path)
    if needs_human:
        entries = [e for e in entries if e.get("needs_human") == "yes"]
    return entries


def _format_human(e):
    failed = sorted(k for k, v in (e.get("gates") or {}).items() if v == "fail")
    tail = ("  fail=%s" % ",".join(failed)) if failed else ""
    return "%s | %s | %s | needs_human=%s | %s%s" % (
        e.get("id"), e.get("card"), e.get("stage"),
        e.get("needs_human"), e.get("claim"), tail)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="inbox.py", description=__doc__)
    parser.add_argument("--inbox", default=str(DEFAULT_INBOX),
                        help="INBOX.jsonl 路径（测试用）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ap = sub.add_parser("append", help="追加一条待审记录")
    ap.add_argument("--card", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--claim", required=True)
    ap.add_argument("--gates", action="append", default=[],
                    help="NAME=pass|fail，可逗号分隔或重复传")
    ap.add_argument("--artifacts", action="append", default=[],
                    help="artifact 路径，可逗号分隔或重复传")
    ap.add_argument("--ts", default=None, help="覆盖 id 时间前缀（测试用）")

    lp = sub.add_parser("list", help="打印待审记录")
    lp.add_argument("--needs-human", action="store_true",
                    help="只显示 needs_human=yes")
    lp.add_argument("--json", action="store_true", help="按 JSONL 输出")

    args = parser.parse_args(argv)
    try:
        if args.cmd == "append":
            gates = parse_gates(args.gates)
            entry = build_entry(args.card, args.stage, args.claim, gates,
                                args.artifacts, args.ts)
            append_entry(args.inbox, entry)
            print("%s  needs_human=%s" % (entry["id"], entry["needs_human"]))
            return 0
        entries = list_entries(args.inbox, args.needs_human)
        for entry in entries:
            if args.json:
                print(json.dumps(entry, ensure_ascii=False))
            else:
                print(_format_human(entry))
        return 0
    except DuplicateId as exc:
        print("拒绝重复追加：%s" % exc, file=sys.stderr)
        return 3
    except ValueError as exc:
        print("拒绝：%s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
