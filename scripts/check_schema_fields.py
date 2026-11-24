#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""加字段前先问一句：这个字段 schema 认不认。

## 为什么有这个脚本

2026-11-24/25 同一个错误犯了 **三次**：
  ① rmsnorm-fused.yaml 加 `comparison_note` → E6 未知字段
  ② 5 张降级卡加 `downgrade_reason`/`downgraded_on` → E6 未知字段
  ③ 同一次里把 `last_verdict.status` 写成 `draft`（枚举里只有 none/predicted/
     measured/refuted/at_floor）→ E6 非法值

三次都是"我以为可以加"，而答案就在 `validate.py` 的常量里。
30 行的检查器比第四次犯错便宜 —— 这是巡检给的规则，本文件实现它。

## 判据从哪来

不自己维护一份白名单（那会变成第三处记账），而是**从 validate.py 的常量读**：
  STATUSES / VERDICT_STATUSES / REQUIRED_TOP / 各子结构允许的键。
读不到就退化成"只检查已知的三个枚举"，并在输出里说明。

用法：
    python3 -I scripts/check_schema_fields.py <卡.yaml> --field <字段名>
    python3 -I scripts/check_schema_fields.py <卡.yaml> --list      # 列出所有合法字段
    python3 -I scripts/check_schema_fields.py --selftest
"""
import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VALIDATE = REPO / "Plugins/ops/op-atlas/validate.py"


def _load_validate():
    """按路径加载 validate.py，只取常量（不跑校验逻辑）。"""
    import importlib.util
    if not VALIDATE.is_file():
        return None, "找不到 %s" % VALIDATE
    spec = importlib.util.spec_from_file_location("_validate_consts", str(VALIDATE))
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except Exception as e:                       # noqa: BLE001
        return None, "加载 validate.py 失败：%s" % e
    return m, None


def _top_keys(mod):
    """卡片允许的顶层字段。从 validate.py 的 REQUIRED_TOP + E6 的 extra 规则反推。

    E6 的实际判据是 `set(c) - ALLOWED_TOP`，所以真正的白名单是 ALLOWED_TOP
    （若不存在则由 `_card_top_keys()` 之类函数给出）。这里做多路查找，
    找不到就返回 None 表示"无法判定"，**不假装通过**。
    """
    for name in ("ALLOWED_TOP", "CARD_KEYS", "TOP_KEYS", "CARD_TOP"):
        v = getattr(mod, name, None)
        if isinstance(v, (set, frozenset, list, tuple)) and v:
            return set(v), name
    # 退化：从 REQUIRED_TOP 只能知道必填，不知道全集
    req = getattr(mod, "REQUIRED_TOP", None)
    if isinstance(req, (set, frozenset, list, tuple)) and req:
        return set(req), "REQUIRED_TOP（只有必填项，非全集）"
    return None, None


def _verdict_keys(mod):
    for name in ("VERDICT_KEYS", "LAST_VERDICT_KEYS"):
        v = getattr(mod, name, None)
        if isinstance(v, (set, frozenset, list, tuple)) and v:
            return set(v), name
    return None, None


def check(mod, card_path, field, parent=None):
    """返回问题列表。空 = 该字段可以加。"""
    probs = []
    top, src = _top_keys(mod)
    if top is None:
        return ["无法从 validate.py 取到卡片顶层白名单 —— 判据失效，请人工查 E6 规则"]
    if parent in (None, "top"):
        if field not in top:
            probs.append(
                "顶层字段 %r 不在白名单（来源 %s）。可选 %s"
                % (field, src, sorted(top)))
        return probs
    if parent == "last_verdict":
        vk, vsrc = _verdict_keys(mod)
        if vk is None:
            return ["取不到 last_verdict 的字段白名单 —— 判据失效"]
        if field not in vk:
            probs.append("last_verdict 字段 %r 不合法（来源 %s）。可选 %s"
                         % (field, vsrc, sorted(vk)))
        return probs
    return ["--parent=%r 不支持（目前只支持 top / last_verdict）" % parent]


def check_enum(mod, field, value):
    """枚举型字段的取值检查。"""
    name = None
    if field in ("status",):
        name, allowed = "STATUSES", getattr(mod, "STATUSES", None)
    elif field in ("verdict_status",) or (field == "status"):
        name, allowed = "VERDICT_STATUSES", getattr(mod, "VERDICT_STATUSES", None)
    if not allowed:
        return None
    return (value in allowed,
            "%s=%r 不在 %s（可选 %s）" % (field, value, name, sorted(allowed)))


def selftest():
    """**反向优先**：先证明它抓得住我犯过的那三次错。"""
    mod, err = _load_validate()
    assert mod is not None, err

    # ① comparison_note —— 真实犯过的错
    p = check(mod, None, "comparison_note")
    assert p and "不在白名单" in p[0], p
    print("[ok] 反向：comparison_note 被拒（2026-11-24 真实犯过）")

    # ② downgrade_reason —— 真实犯过的错
    p = check(mod, None, "downgrade_reason")
    assert p and "不在白名单" in p[0], p
    print("[ok] 反向：downgrade_reason 被拒（2026-11-25 真实犯过）")

    # ③ last_verdict.status='draft' —— 合法字段 + 非法值
    ok, msg = check_enum(mod, "verdict_status", "draft")
    assert ok is False and "VERDICT_STATUSES" in msg, (ok, msg)
    print("[ok] 反向：last_verdict.status='draft' 被拒（枚举里只有 none/predicted/"
          "measured/refuted/at_floor）")

    # 正向：合法字段要通过，否则上面三条是假绿
    p = check(mod, None, "math")
    assert p == [], p
    print("[ok] 正向：math 通过（证明判据不是一律拒绝）")
    ok, _ = check_enum(mod, "verdict_status", "none")
    assert ok is True
    print("[ok] 正向：last_verdict.status='none' 通过")
    print("selftest PASS")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("card", nargs="?", help="卡 yaml（仅用于显示，判据不依赖它）")
    ap.add_argument("--field", help="要加/要检查的字段名")
    ap.add_argument("--parent", default="top",
                    help="字段挂在哪：top（默认）或 last_verdict")
    ap.add_argument("--value", help="枚举字段的取值（如 status=draft）")
    ap.add_argument("--list", action="store_true", help="列出合法字段")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    mod, err = _load_validate()
    if err:
        print(err)
        return 2
    if a.list:
        top, src = _top_keys(mod)
        vk, vsrc = _verdict_keys(mod)
        print("卡片顶层（%s）：%s" % (src, sorted(top) if top else "取不到"))
        print("last_verdict（%s）：%s" % (vsrc, sorted(vk) if vk else "取不到"))
        print("STATUSES：%s" % sorted(getattr(mod, "STATUSES", [])))
        print("VERDICT_STATUSES：%s" % sorted(getattr(mod, "VERDICT_STATUSES", [])))
        return 0
    if not a.field:
        ap.error("需要 --field 或 --list 或 --selftest")
    probs = check(mod, a.card, a.field, a.parent)
    if a.value:
        r = check_enum(mod, "verdict_status" if a.parent == "last_verdict" else "status",
                       a.value)
        if r and r[0] is False:
            probs.append(r[1])
    for p in probs:
        print("✗ " + p)
    if not probs:
        print("✓ %r 可以加（parent=%s）" % (a.field, a.parent))
    return 0 if not probs else 1


if __name__ == "__main__":
    sys.exit(main())
