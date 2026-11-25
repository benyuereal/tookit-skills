#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_s_direction —— 找出「S 项满分但整体没变快」的卡。

## 为什么需要它

`depth.py` 的 `shape_score()` 判据是：

    if len(shapes) >= 3 and gm.get("value") is not None:
        return 1.0

**只看形状个数，不看方向。** geomean 是 0.5（快一倍）还是 1.45（慢 45%），
给的分完全一样。

后果：凑够三个形状就能拿满 S 的 20 分，哪怕每个形状都更慢。
E4「十张满卡」的目标数字会因此虚高。

## 这个脚本做什么（以及不做什么）

它**只报告，不改分**。原因：改 `shape_score` 是口径变更，
会让所有历史分数不可比，需要单独决策。

它做的事是把问题变成可枚举的清单，而不是停留在某次汇报的口头讨论里。

## 用法

    python3 -I scripts/check_s_direction.py          # 列出所有嫌疑卡
    python3 -I scripts/check_s_direction.py --strict # 有嫌疑卡则退出码 1
    python3 -I scripts/check_s_direction.py --selftest
"""
import argparse
import pathlib
import sys

CARDS = pathlib.Path(__file__).resolve().parents[1] / "Plugins/ops/op-atlas/cards"
TOL = 0.98   # geomean 略小于 1 也算「基本没变快」，给 2% 余量


def scan(cards_dir=None):
    """返回 [(card_id, geomean, n_shapes)] —— 满足「S 满分但没变快」。"""
    import yaml
    d = pathlib.Path(cards_dir) if cards_dir else CARDS
    out = []
    for p in sorted(d.glob("*.yaml")):
        c = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        lv = c.get("last_verdict") or {}
        gm = lv.get("geomean") or {}
        val, shapes = gm.get("value"), gm.get("shapes") or []
        if not isinstance(val, (int, float)):
            continue
        if len(shapes) >= 3 and val >= TOL:
            out.append((c.get("id") or p.stem, float(val), len(shapes)))
    return out


def selftest():
    import tempfile
    import yaml
    with tempfile.TemporaryDirectory() as td:
        d = pathlib.Path(td)

        def mk(name, val, n):
            shares = [{"id": name, "last_verdict": {"geomean": {
                "value": val, "shapes": ["s%d" % i for i in range(n)]}}}]
            (d / ("%s.yaml" % name)).write_text(
                yaml.safe_dump(shares[0], allow_unicode=True), encoding="utf-8")

        mk("good_fast", 0.50, 3)      # 快 -> 不该报
        mk("bad_slow", 1.45, 3)       # 慢 -> 该报
        mk("two_shapes", 1.80, 2)     # 只有 2 形状，S 本来就不满分 -> 不报
        mk("no_geomean", None, 3)     # 无 value -> 不报

        hits = scan(d)
        names = {h[0] for h in hits}
        assert names == {"bad_slow"}, names
        print("[ok] 只报「3 形状 + geomean>=0.98」的卡：%s" % sorted(names))

        (d / "bad_slow.yaml").unlink()
        assert scan(d) == [], "清掉后应为空"
        print("[ok] 清掉嫌疑卡后返回空")

    print("\nselftest PASS（2 项）")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strict", action="store_true",
                    help="有嫌疑卡则退出码 1（给 CI 用）")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()

    hits = scan()
    if not hits:
        print("✓ 没有「S 满分但整体没变快」的卡")
        return 0

    print("以下卡的 S 项拿满分（覆盖 3+ 形状），但 geomean >= %.2f，" % TOL)
    print("即整体并没有比基线快：\n")
    for cid, val, n in hits:
        print("  %-28s geomean=%.4f  shapes=%d" % (cid, val, n))
    print("\n这不是说数据造假 —— 数字都是真的。")
    print("是 shape_score 只看形状个数、不看方向造成的口径问题。")
    print("要修需改 scripts/depth.py 的 shape_score，属口径变更，需单独决策。")
    return 1 if a.strict else 0


if __name__ == "__main__":
    sys.exit(main())
