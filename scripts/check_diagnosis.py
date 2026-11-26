#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断记录必须带「假设 → 判定」四字段。

## 由来

2026-11-25 的一次调试里，一天出了 4 次结论，前 3 次全被推翻：

    ① d=512 不支持         依据：代码半截图        → **被推翻**
    ② H>=64 未初始化指针    依据：读过代码          → **被推翻**
    ③ HCU[4-7] 平台问题     依据：换卡崩            → **被推翻**
    ④ cuda:N 用法问题       依据：最小复现 + 对照    → 正确

问题不在「想得不够」，在于**每次都把当时的最优假设写成了「结案」「最终结论」**，
却没写「什么实测结果能推翻它」。于是下一条相反结论来时，上一条还挂着「结案」，
没人知道哪条还算数。commit 标题本身就是证据：
`2f22f6f docs(D14): 结案 —— …`、`cf5d49a docs(D24): 最终结论 —— …`。

正确形态是「当前最优假设 + 可观测的 falsifier」：结论能带一个判据，
下一次实测就能自动判它是对是错，而不是靠人记得。

## 诊断记录落在哪（本文件的扫描范围）

   落点                                 谁写的
   ──────────────────────────────────   ───────────────────────────────
   <card>/runs/loop/**/*.jsonl          engine/loop.py::append_record()
   <card>/candidates.yaml:refuted[]     engine/loop.py::_to_refuted()
   卡片 last_verdict.next_probe         validate.py E26（measured/tuned 必填）

三处都是「一条关于某假设/结论的记录」，但它们各自的字段名不同（`why` /
`kind` / `root_cause`），没有一个统一的 machine-checkable 契约。
本文件定义这个契约，并把它变成能跑的命令。

## 一条诊断记录必须含

    hypothesis   假设是什么（一句话，非空）
    falsifier    什么实测结果能证明它错（**必须可观测**；不能是「看看」「待定」）
    evidence     支持它的实测证据（status=已证 时必填）
    status       未验 / 部分证实 / 已证 / 已被推翻  （四枚举）

task 原文把 status 注成三个值、判据又写「四个枚举」；这里取四个，
把「部分证实」单列出来 —— 有支持证据但未排除反例，和「已证」不是一回事，
混在一起正是「半截图就被当成已证」的源头。

## 用法

    python3 -I scripts/check_diagnosis.py            # 扫全仓，列出不合规
    python3 -I scripts/check_diagnosis.py --strict   # 有不合规则退出码 1
    python3 -I scripts/check_diagnosis.py --json
    python3 -I scripts/check_diagnosis.py --selftest

**本文件只报告，不改记录。** 存量记录写于本契约之前，批量补字段等于编造
falsifier。先让问题可见（契约测试用 warnings.warn 输出清单），再按卡逐个补。
"""
import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]

# status 的四个枚举值。任何一处判 status 都引用这里，不另抄一份。
STATUSES = ("未验", "部分证实", "已证", "已被推翻")

# falsifier 显然不算判据的写法。比较时先去掉首尾标点。
VAGUE = {"看看", "待定", "再说", "待查", "无", "未知", "todo", "tbd", "n/a",
         "?", "？", "不是", "不知道", "不清楚"}
MIN_FALSIFIER_LEN = 4   # 「看看」「待定」这类两字词挡不住就再加长度线


def _s(v):
    """取字符串并 strip；非字符串（list/dict）返回空 —— 这四个字段都该是标量。"""
    return v.strip() if isinstance(v, str) else ""


def _has(v):
    """evidence 可以是字符串或列表：都按「非空」算。"""
    if v is None:
        return False
    if isinstance(v, str):
        return bool(v.strip())
    if isinstance(v, (list, tuple, dict)):
        return len(v) > 0
    return bool(v)


def check_record(rec, where=""):
    """校验一条诊断记录。返回问题清单；空 = 合规。

    `where` 只用于错误信息定位，不参与判定。
    """
    if not isinstance(rec, dict):
        return ["记录不是 mapping（%s）" % (where or "?")]
    probs = []

    if not _s(rec.get("hypothesis")):
        probs.append("缺 hypothesis（假设是什么，一句话）")

    fal = _s(rec.get("falsifier"))
    if not fal:
        probs.append("缺 falsifier（什么实测结果能证明它错）")
    else:
        core = fal.strip(" 。.！!；;，,").lower()
        if core in VAGUE or len(core) < MIN_FALSIFIER_LEN:
            probs.append(
                "falsifier 是空话 %r —— 必须写成可观测的实测判据（例："
                "「若 d=512 在 HCU[0] 上跑通且输出正确，则本假设被推翻」）" % fal)

    st = _s(rec.get("status"))
    if not st:
        probs.append("缺 status（%s）" % " / ".join(STATUSES))
    elif st not in STATUSES:
        probs.append("status=%r 不在 %s" % (st, " / ".join(STATUSES)))

    if st == "已证" and not _has(rec.get("evidence")):
        probs.append("status=已证 但 evidence 为空 —— 已证必须有实测证据")

    return probs


# ---------------------------------------------------------------- 扫描落点

def _iter_loop_jsonl(repo):
    """runs/loop/**/*.jsonl —— 每条带 decision 的记录都是一次判定。"""
    skills = repo / "Plugins/ops/skills"
    for f in sorted(skills.glob("*/runs/loop/**/*.jsonl")):
        try:
            text = f.read_text(encoding="utf-8")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict) and "decision" in rec:
                where = "%s:%d" % (f.relative_to(repo), i)
                yield where, "runs/loop", rec


def _iter_refuted(repo):
    """candidates.yaml:refuted[] —— 「已判死」的结论。"""
    try:
        import yaml
    except ImportError:                                        # pragma: no cover
        return
    for p in sorted((repo / "Plugins/ops/skills").glob("*/candidates.yaml")):
        try:
            d = yaml.safe_load(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(d, dict):
            continue
        for j, r in enumerate(d.get("refuted") or []):
            if isinstance(r, dict):
                yield "%s:refuted[%d]" % (p.relative_to(repo), j), "refuted", r


def _iter_next_probe(repo):
    """卡片 last_verdict.next_probe —— 前向假设（已有 hypothesis/falsifier 两字段）。"""
    try:
        import yaml
    except ImportError:                                        # pragma: no cover
        return
    for p in sorted((repo / "Plugins/ops/op-atlas/cards").glob("*.yaml")):
        try:
            d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except (OSError, ValueError):
            continue
        lv = d.get("last_verdict") or {}
        np_ = lv.get("next_probe") if isinstance(lv, dict) else None
        if isinstance(np_, dict):
            yield "%s:last_verdict.next_probe" % p.relative_to(repo), "next_probe", np_


CARRIERS = (_iter_loop_jsonl, _iter_refuted, _iter_next_probe)


def scan(repo=None):
    """扫全仓诊断记录。返回 [{"where", "source", "problems"}]（只含不合规项）。"""
    repo = pathlib.Path(repo) if repo else REPO
    bad = []
    for it in CARRIERS:
        for where, source, rec in it(repo):
            probs = check_record(rec, where)
            if probs:
                bad.append({"where": where, "source": source, "problems": probs})
    return bad


def counts(repo=None):
    """按落点统计 (总数, 不合规数)。"""
    repo = pathlib.Path(repo) if repo else REPO
    out = {}
    for it in CARRIERS:
        src = None
        for where, source, rec in it(repo):
            src = source
            tot, bad = out.get(source, (0, 0))
            out[source] = (tot + 1, bad + (1 if check_record(rec, where) else 0))
        if src is not None:
            out.setdefault(src, (0, 0))
    return out


# ---------------------------------------------------------------- selftest

def selftest(quiet=False):
    """正向 + 反向用例。**反向用例必须真的会失败**，否则这就是假绿。"""
    import tempfile

    def say(m):
        if not quiet:
            print(m)

    n = 0

    good = {
        "hypothesis": "d=512 不支持",
        "falsifier": "若 d=512 的 kernel 在 HCU[0] 上跑通且输出与参考一致，则本假设被推翻",
        "evidence": "代码半截图：dense_decode.h 无 d=512 分支",
        "status": "已证",
    }
    assert check_record(good) == [], check_record(good)
    say("[ok] 正向：四字段齐全、status=已证 带证据 → 通过")
    n += 1

    # 正向：未验可以没有 evidence
    assert check_record({"hypothesis": "x", "falsifier": "跑通即推翻，判据见 bench",
                         "evidence": "", "status": "未验"}) == []
    say("[ok] 正向：status=未验 允许 evidence 为空")
    n += 1

    # 反向：falsifier 为空必须报错（task 明确要求）
    p = check_record({"hypothesis": "x", "falsifier": "",
                      "evidence": "e", "status": "已证"})
    assert any("falsifier" in x for x in p), p
    say("[ok] 反向：falsifier 为空 → 报错")
    n += 1

    # 反向：falsifier 是空话
    p = check_record({"hypothesis": "x", "falsifier": "看看",
                      "evidence": "e", "status": "已证"})
    assert any("空话" in x for x in p), p
    say("[ok] 反向：falsifier=「看看」→ 报空话")
    n += 1

    # 反向：status 不在四枚举
    p = check_record({"hypothesis": "x", "falsifier": "跑通即推翻，判据见 bench",
                      "evidence": "e", "status": "已确认"})
    assert any("status" in x for x in p), p
    say("[ok] 反向：status=「已确认」→ 报不在枚举")
    n += 1

    # 反向：已证但 evidence 空
    p = check_record({"hypothesis": "x", "falsifier": "跑通即推翻，判据见 bench",
                      "evidence": "", "status": "已证"})
    assert any("evidence" in x for x in p), p
    say("[ok] 反向：status=已证 但 evidence 为空 → 报错")
    n += 1

    # 反向：空记录四个字段全缺
    p = check_record({})
    assert len(p) >= 3, p
    say("[ok] 反向：空记录 → 至少 3 项缺失")
    n += 1

    # 扫描落点：临时仓里放 1 条合规 + 2 条不合规，scan 只应报 2
    try:
        import yaml  # noqa: F401
        has_yaml = True
    except ImportError:                                    # pragma: no cover
        has_yaml = False
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        card = root / "Plugins/ops/skills/op-probe"
        (card / "runs/loop").mkdir(parents=True)
        good_line = json.dumps({"decision": "ACCEPTED_NUMERICS", **good},
                               ensure_ascii=False)
        bad_line = json.dumps({"decision": "REJECTED_NO_GAIN",
                               "hypothesis": "y"}, ensure_ascii=False)
        (card / "runs/loop/20260101.jsonl").write_text(
            good_line + "\n" + bad_line + "\n", encoding="utf-8")
        if has_yaml:
            (root / "Plugins/ops/op-atlas/cards").mkdir(parents=True)
            (root / "Plugins/ops/op-atlas/cards/probe.yaml").write_text(
                "last_verdict:\n  next_probe:\n    hypothesis: 融合应省一趟\n"
                "    falsifier: 看看\n", encoding="utf-8")
        bad = scan(root)
        # jsonl: 1 条不合规；next_probe: 1 条不合规（若 yaml 可用）
        expect = 2 if has_yaml else 1
        assert len(bad) == expect, (len(bad), bad)
        say("[ok] 扫描：临时仓 %d 条不合规，合规记录未被误报" % expect)
        n += 1

    say("\nselftest PASS（%d 项）" % n)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=str(REPO), help="仓库根（默认本仓）")
    ap.add_argument("--strict", action="store_true", help="有不合规则退出码 1")
    ap.add_argument("--json", action="store_true", help="机器可读输出")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    bad = scan(a.repo)
    cnt = counts(a.repo)
    if a.json:
        print(json.dumps({"counts": cnt, "noncompliant": bad},
                         ensure_ascii=False, indent=2))
        return 1 if (a.strict and bad) else 0

    total = sum(t for t, _ in cnt.values())
    nbad = sum(b for _, b in cnt.values())
    print("诊断记录四字段契约（hypothesis / falsifier / evidence / status）")
    print("扫到 %d 条，%d 条不合规\n" % (total, nbad))
    for src in ("runs/loop", "refuted", "next_probe"):
        if src in cnt:
            t, b = cnt[src]
            print("  %-12s %3d/%3d 不合规" % (src, b, t))
    if not bad:
        print("\n✓ 全部合规")
        return 0

    print("\n不合规清单（前 200 条）：")
    for item in bad[:200]:
        print("  ✗ %s" % item["where"])
        for p in item["problems"]:
            print("      - %s" % p)
    if len(bad) > 200:
        print("  … 其余 %d 条省略（--json 看全量）" % (len(bad) - 200))
    print("\n存量记录写于本契约之前，不批量补 —— 补不出真实 falsifier 就是编造。")
    return 1 if a.strict else 0


if __name__ == "__main__":
    sys.exit(main())
