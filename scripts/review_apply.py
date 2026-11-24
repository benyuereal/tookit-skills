#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把人的审阅结果落到卡片 progress —— **人只有降级权，没有提权**。

设计依据：`/root/atlas-handoff/WORKFLOW-UPGRADE-PLAN.md` §3.4（业主裁定）。

输入：docs/REVIEWS.jsonl（人写，append-only）
输出：cards/<card>.yaml 的 progress 字段

`action` 只允许三个值：

  ack        认可。**一行也不改卡片**（只留记录）。
  downgrade  降级。只允许把 progress.stage **调低**，或设 progress.blocked_by。
  suspect    标疑。只加顶层 `suspect: true` 标记，**不碰任何门禁结论**。

三条写死的判据（任何一条命中 → 拒绝该行、打印「人工无提权」、退出码非 0）：

  ① `to` 指向的阶段 index >= 当前阶段 index（S4→S7、S4→S4 都不算降级）；
  ② 行里出现任何 gate 相关字段（gates / gate / clear_gates / set_gates /
     override / force / E4x …）——那是「把 fail 改成 pass」的入口；
  ③ ack 行携带 `to` 或 `blocked_by`——ack 只记录，不许改。

结构性保证：`validate.py` **不读** REVIEWS.jsonl。人的批准不进任何 E 判据，
所以即使本脚本被绕过，也提不了权。

用法：
  python3 -I scripts/review_apply.py                 # 应用到全部卡
  python3 -I scripts/review_apply.py --dry-run       # 只报告不落盘
"""
import argparse
import json
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_REVIEWS = REPO / "docs/REVIEWS.jsonl"
DEFAULT_CARDS = REPO / "Plugins/ops/op-atlas/cards"

# 与 validate.py 的 STAGES / BLOCKED_BY 保持一致（只读引用，不改它）。
STAGES = ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7")
BLOCKED_BY = {"semantics", "no_measured_time", "no_baseline",
              "no_counters", "no_resources", "no_evidence"}
ACTIONS = {"ack", "downgrade", "suspect"}

# 审阅行允许的字段。少一个 key 都拒绝——多出来的 key 就是提权入口。
ALLOWED_REVIEW_KEYS = {"id", "reviewer", "action", "to", "blocked_by",
                       "reason", "ts", "card"}
# 一看到这些 key / 前缀，直接判「试图改门禁」。
FORBIDDEN_PREFIXES = ("gate", "clear_", "set_gate", "override", "force",
                      "e41", "e42", "e43", "e39", "e40")

NO_ESCALATION = "人工无提权"


class ReviewRejected(Exception):
    """该行被拒绝。message 里一定带「人工无提权」或具体原因。"""


def _err(msg):
    raise ReviewRejected(msg)


def _stage_index(stage):
    return STAGES.index(stage) if stage in STAGES else None


def _forbidden_keys(review):
    bad = []
    for key in review:
        low = key.lower()
        if low in {"to", "blocked_by", "id", "reviewer", "action",
                   "reason", "ts", "card"}:
            continue
        if any(low == p or low.startswith(p) for p in FORBIDDEN_PREFIXES):
            bad.append(key)
    return bad


CARD_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def card_id_from_review(review):
    """优先取显式 `card`；否则从 id `<ts>/<card>/<stage>` 里取倒数第二段。"""
    if not isinstance(review, dict):
        _err("审阅行必须是 JSON object")
    card = review.get("card")
    if card:
        name = str(card)
    else:
        rid = str(review.get("id") or "")
        parts = rid.split("/")
        if len(parts) < 3:
            _err("id %r 无法解析出 card（应为 <ts>/<card>/<stage>）" % rid)
        name = parts[-2]
    # 卡名会拼进文件路径，必须挡住 ../ 与绝对路径。
    if not CARD_NAME_RE.match(name) or name in (".", ".."):
        _err("card 名 %r 非法（只允许字母数字 . _ -）" % name)
    return name


def validate_review(review, card=None):
    """对单行做**与卡片无关**的合法性检查。返回 (action, to, blocked_by)。

    这里拒绝的是「格式/动作本身就该拒」的行；与当前 stage 的比较在 plan_apply。
    """
    if not isinstance(review, dict):
        _err("审阅行必须是 JSON object")

    bad = _forbidden_keys(review)
    if bad:
        _err("%s：审阅行带门禁字段 %s —— 人的批准不能改 gate 结论"
             % (NO_ESCALATION, bad))

    unknown = set(review) - ALLOWED_REVIEW_KEYS - set(bad)
    if unknown:
        # 审阅行只允许认几个字段；多出来的字段没有合法用途，
        # 只可能是想顺手改 gate 或卡片其它字段——一律按提权处理。
        _err("%s：审阅行带未授权字段 %s（只接受 %s）"
             % (NO_ESCALATION, sorted(unknown), sorted(ALLOWED_REVIEW_KEYS)))

    action = review.get("action")
    if action not in ACTIONS:
        _err("action=%r 非法（只允许 ack/downgrade/suspect）" % (action,))
    if not str(review.get("reviewer") or "").strip():
        _err("缺 reviewer")

    to = review.get("to")
    blocked_by = review.get("blocked_by")

    if action == "ack":
        # ack = 只记录。带任何改动字段都说明它想顺手改点什么。
        if to is not None or blocked_by is not None:
            _err("%s：ack 只记录，不许带 to/blocked_by" % NO_ESCALATION)
        return action, None, None

    if action == "suspect":
        if to is not None:
            _err("%s：suspect 只加标记，不许带 to" % NO_ESCALATION)
        if blocked_by is not None:
            _err("suspect 不许带 blocked_by（那属于 downgrade）")
        if not str(review.get("reason") or "").strip():
            _err("suspect 必须写 reason（标疑要有理由）")
        return action, None, None

    # ---- downgrade ----
    if to is None and blocked_by is None:
        _err("downgrade 必须给 to（阶段）或 blocked_by")
    if blocked_by is not None and blocked_by not in BLOCKED_BY:
        _err("blocked_by=%r 非法（只允许 %s）"
             % (blocked_by, sorted(BLOCKED_BY)))
    if to is not None and to != "blocked" and to not in STAGES:
        _err("to=%r 非法（只允许 S0..S7 或 'blocked'）" % (to,))
    if to == "blocked" and blocked_by is None:
        _err("to=blocked 必须同时给 blocked_by（阻塞理由不能空着）")
    if not str(review.get("reason") or "").strip():
        _err("downgrade 必须写 reason")
    return action, to, blocked_by


def _progress_of(card):
    p = card.get("progress")
    if p is None:
        return {}
    if not isinstance(p, dict):
        _err("卡片的 progress 不是 mapping")
    return p


def plan_apply(review, card):
    """把一行审阅换算成对卡片的修改。返回 (new_card, note)。

    **提权判据就在这个函数里**：任何「往上走」的分支都抛 ReviewRejected。
    幂等：同一行跑第二遍时 stage 已经等于 to，视作已应用，不再报错。
    """
    action, to, blocked_by = validate_review(review, card)
    card = dict(card)                       # 不改调用方对象
    progress = dict(_progress_of(card))
    rid = review.get("id")

    if action == "ack":
        return card, "ack 不改字段"

    if action == "suspect":
        card["suspect"] = True
        return card, "suspect=true"

    # ---- downgrade ----
    cur = progress.get("stage")
    cur_idx = _stage_index(cur)

    if to is not None and to != "blocked":
        new_idx = _stage_index(to)
        if cur_idx is None:
            # 卡上还没有合法 stage：写入任何阶段都等于新开一个状态，先拒绝。
            _err("%s：卡片当前 stage=%r 不是合法阶段，拒绝由人工设定"
                 % (NO_ESCALATION, cur))
        if new_idx > cur_idx:
            # 核心判据：阶段只能往下。相等 (S4->S4) 视作「已应用」，保证幂等。
            _err("%s：%s 试图把 stage 从 %s 提到 %s（只许降）"
                 % (NO_ESCALATION, rid, cur, to))
        progress["stage"] = to
        progress.setdefault("since", _now_iso())
        note = "stage %s -> %s" % (cur, to)
        if blocked_by is not None:
            progress["blocked_by"] = blocked_by
            note += "，blocked_by=%s" % blocked_by
    else:
        # to == "blocked" 或只给 blocked_by
        if cur is None:
            _err("卡片当前没有合法 stage，无法只设 blocked_by")
        progress["blocked_by"] = blocked_by
        progress.setdefault("since", _now_iso())
        note = "blocked_by=%s（stage 保持 %s）" % (blocked_by, cur)

    card["progress"] = progress
    return card, note


def _now_iso():
    return datetime.now().replace(microsecond=0).isoformat()


def _load_card(path):
    """读卡。文件可能不存在 → ReviewRejected（不能把审阅落进空气）。"""
    if not path.is_file():
        _err("卡片文件不存在：%s" % path)
    try:
        import yaml
    except ImportError:  # pragma: no cover - 环境缺 yaml 属于硬故障
        _err("需要 PyYAML 才能读卡片")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        _err("卡片顶层不是 mapping：%s" % path)
    return data


def _dump_card(path, card):
    import yaml
    text = yaml.dump(card, allow_unicode=True, sort_keys=False, width=200)
    # 原子写：先在同目录写临时文件，再 rename；避免半截卡片。
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with open(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        Path(tmp).replace(path)
    finally:
        if Path(tmp).exists():
            Path(tmp).unlink()


def read_reviews(path):
    """读 REVIEWS.jsonl。不存在 = 空（人还没审过不是错误）。"""
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            _err("%s 第 %d 行不是合法 JSON：%s" % (path, lineno, exc))
        rows.append((lineno, obj))
    return rows


def apply_reviews(reviews_path=DEFAULT_REVIEWS, cards_dir=DEFAULT_CARDS,
                  dry_run=False):
    """把 REVIEWS 全部应用。返回报告 dict：

      applied   [(lineno, id, note)]
      rejected  [(lineno, id, msg)]      ← 含「人工无提权」
      skipped   [(lineno, id, msg)]      ← 卡片不存在等
      ok        bool（没有 rejected/skipped 且无读文件错误）
    """
    reviews_path = Path(reviews_path)
    cards_dir = Path(cards_dir)
    report = {"applied": [], "rejected": [], "skipped": [], "ok": True}

    try:
        rows = read_reviews(reviews_path)
    except ReviewRejected as exc:
        report["rejected"].append((0, None, str(exc)))
        report["ok"] = False
        return report

    # REVIEWS append-only：本脚本从头到尾只以 "r" 打开它。
    cards_cache = {}
    for lineno, review in rows:
        rid = review.get("id") if isinstance(review, dict) else None
        try:
            card_name = card_id_from_review(review)
        except ReviewRejected as exc:
            report["rejected"].append((lineno, rid, str(exc)))
            continue
        card_path = cards_dir / ("%s.yaml" % card_name)
        try:
            if card_name not in cards_cache:
                cards_cache[card_name] = (_load_card(card_path), card_path)
            card, path = cards_cache[card_name]
            new_card, note = plan_apply(review, card)
            if new_card != card:
                if not dry_run:
                    _dump_card(path, new_card)
                cards_cache[card_name] = (new_card, path)
                report["applied"].append((lineno, rid, note))
            else:
                report["applied"].append((lineno, rid, note + "（无变化）"))
        except ReviewRejected as exc:
            msg = str(exc)
            if card_path.is_file() or NO_ESCALATION in msg:
                report["rejected"].append((lineno, rid, msg))
            else:
                report["skipped"].append((lineno, rid, msg))

    report["ok"] = not report["rejected"] and not report["skipped"]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog="review_apply.py", description=__doc__)
    parser.add_argument("--reviews", default=str(DEFAULT_REVIEWS))
    parser.add_argument("--cards", default=str(DEFAULT_CARDS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    report = apply_reviews(args.reviews, args.cards, args.dry_run)
    for lineno, rid, note in report["applied"]:
        print("应用 %s:%s  %s" % (lineno, rid, note))
    for lineno, rid, msg in report["skipped"]:
        print("跳过 %s:%s  %s" % (lineno, rid, msg), file=sys.stderr)
    for lineno, rid, msg in report["rejected"]:
        print("拒绝 %s:%s  %s" % (lineno, rid, msg), file=sys.stderr)
    print("共 %d 行：应用 %d / 跳过 %d / 拒绝 %d"
          % (sum(len(report[k]) for k in ("applied", "skipped", "rejected")),
             len(report["applied"]), len(report["skipped"]),
             len(report["rejected"])))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
