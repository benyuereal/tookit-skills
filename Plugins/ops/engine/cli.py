# -*- coding: utf-8 -*-
"""cli —— explore / floor 的**薄 CLI 层**：只解析参数、打印、定退出码。

## 薄到什么程度

本模块允许出现：argparse、print、json.dumps、写卡落盘、退出码。
本模块**不允许**出现：判定口径、地板公式、候选规则 —— 那些在
`floor.py` / `probe.py` / `propose.py` 里。

## 为什么 `scripts/explore.py` 不直接写这些

因为它还要能被 `python3 -I` 跑（脚本目录不进 sys.path）、被 importlib 按路径
加载（测试）、被别的脚本当模块调。把「参数与打印」放在这里、把「怎么算」放在
engine 里，三种加载方式就都只需要认识**一个**入口（`main_*`）。

## 为什么 advance 的 main 不在这里

`scripts/advance.py` 的 `run_advance` 读模块级 `SURFACES_PATH`，而测试用
monkeypatch 改的正是那个名字（见 test_advance 的 surfaces_file 夹具）。
把它搬进 engine 会让补丁静默失效 —— 测试仍然"通过"，但测的东西已经不是
真实路径了。所以编排骨架留在原文件，只把纯逻辑搬走。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ENGINE_DIR = Path(__file__).resolve().parent


def _engine(name):
    """按路径加载 engine/ 下的兄弟模块（见 engine/loop.py::_engine 的说明）。"""
    key = "_ops_engine_" + name
    if key in sys.modules:
        return sys.modules[key]
    import importlib.util
    spec = importlib.util.spec_from_file_location(key, str(_ENGINE_DIR / (name + ".py")))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- explore

def explore_report(card, shape=None, limit=5):
    """采信号 → 出题 → 截断。返回 {"signals", "proposals", "skipped"}。

    这是 `explore` 这个**命令**的语义，不是「引擎」的语义：换一个前端
    （比如 loop.py 的 next_step_hint）可以要 signals 但不要截断。
    """
    probe = _engine("probe")
    propose = _engine("propose")
    shape = probe.pick_shape(card, shape)
    if not shape:
        return {"signals": {}, "proposals": [],
                "skipped": ["shapes.yaml 为空，无档位可选"]}
    sig = probe.collect(card, shape)
    props, skipped = propose.propose(sig)
    return {"signals": sig, "proposals": props[:limit],
            "skipped": skipped + (props[limit:] and ["超出 limit %d，未列出" % limit] or [])}


def build_explore_parser():
    ap = argparse.ArgumentParser(description="从通用信号出候选（拟做，不是待做）")
    ap.add_argument("--card", required=True)
    ap.add_argument("--shape", default=None, help="shapes.yaml 的档位名；缺省自动选")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--apply", action="store_true", help="写 runs/advance/overlay.yaml")
    ap.add_argument("--json", action="store_true")
    return ap


def main_explore(argv=None):
    probe = _engine("probe")
    propose = _engine("propose")
    ledger = _engine("ledger")
    import json
    a = build_explore_parser().parse_args(argv)

    r = explore_report(a.card, a.shape, a.limit)
    if a.json:
        print(json.dumps(r, ensure_ascii=False, indent=2))
    else:
        print(propose.render(r["signals"], r["proposals"], r["skipped"]))
    if a.apply:
        if not r["proposals"]:
            print("\n没有候选可写，不改 overlay。", file=sys.stderr)
            return 1
        # 写 overlay 的实现在 engine/ledger.py（与 advance 的 forge 同一个函数）：
        # 原先这里要 importlib 加载 scripts/advance.py 才能拿到它。
        f = ledger.write_overlay(a.card, r["proposals"])
        print("\n已写 %s（%d 条，status=proposed）" % (f, len(r["proposals"])))
        print("下一步：实现变体 → 把 status 改成 pending 并补 variant/done_when → "
              "`loop.py -C %s run <id>`" % a.card)
    return 0


# ---------------------------------------------------------------- floor

def write_card(card_path, result, spec):
    """把 floor 结果写进卡片。**没有 basis 就不写** —— 这是反作弊线。

    只有数字没有依据的，等于把「算出来的」和「拍脑袋的」混在一起。
    这是 CLI 的副作用（`--write-card`），所以它在 cli 层而不是 floor 计算层。
    """
    import time
    import yaml
    card_io = _engine("card")
    if not result.get("basis"):
        return False, "缺 basis，拒写"
    if result.get("remaining") is None:
        return False, "remaining 为 None，拒写"
    d = card_io.read_yaml(card_path, {})
    lv = d.setdefault("last_verdict", {})
    lv["floor"] = {
        "remaining": result["remaining"],
        "t_floor_us": result["t_floor_us"],
        "t_actual_us": result["t_actual_us"],
        "binding": result["binding"],
        "basis": spec["basis"] + " ⇒ " + result["basis"],
        "verdict": result["gate"],
        "denominator_source": result["denominator_source"],
        "updated": time.strftime("%Y-%m-%d"),
    }
    # E40 只看这两个数；没有就不写（缺数据不等于 0）
    for k in ("achieved_gbs", "achieved_tf"):
        if result.get(k):
            lv["floor"][k] = result[k]
    if result.get("achieved_tf") and spec.get("precision"):
        lv["floor"]["precision"] = spec["precision"]   # fp8/bf16 上限不同，判定要分档
    card_path.write_text(yaml.dump(d, allow_unicode=True, sort_keys=False, width=200),
                         encoding="utf-8")
    return True, "ok"


def measured_cards(cards_dir):
    """atlas 里 status=measured/tuned 的卡 id（`--all` 的作用域）。"""
    card_io = _engine("card")
    out = []
    for p in sorted(Path(cards_dir).glob("*.yaml")):
        d = card_io.read_yaml(p, {})
        if d.get("status") in ("measured", "tuned"):
            out.append(p.stem)
    return out


def print_floor(r, as_json):
    """人读 / JSON 两种打印。JSON 分支输出的是 analyze() 的原始 dict。"""
    if "error" in r:
        print("ERROR: %s" % r["error"], file=sys.stderr)
        return
    if as_json:
        import json
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return
    print("binding    : %s（%s）" % (r["binding"], r["denominator_source"]))
    print("basis      : %s" % r["basis"])
    print("t_floor    : %.1f us" % r["t_floor_us"])
    print("t_actual   : %s us" % (r["t_actual_us"] if r["t_actual_us"] else "?"))
    if r["remaining"] is None:
        print("remaining  : ?")
    else:
        print("remaining  : %+.1f%%  → %s" % (r["remaining"] * 100, r["gate"]))


def build_floor_parser():
    ap = argparse.ArgumentParser(description="算本卡地板与剩余空间")
    ap.add_argument("--card")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--write-card", action="store_true", dest="write_card",
                    help="把 remaining/basis/t_floor_us 写进卡片的 last_verdict.floor")
    ap.add_argument("--all", action="store_true",
                    help="对所有 measured/tuned 卡跑一遍（配 --write-card 批量写）")
    ap.add_argument("--selftest", action="store_true")
    return ap


def main_floor(argv=None):
    floor_io = _engine("floor")
    ap = build_floor_parser()
    a = ap.parse_args(argv)
    if a.selftest:
        return floor_io.selftest()

    # 先校验分母来源齐不齐：缺 bw-ladder 是**环境**坏了，不是「这张卡算不出」，
    # 要像以前一样立刻停（否则 --all 会打几十行 SKIP，让人以为卡片有问题）。
    floor_io.load_ladder()
    cards = (measured_cards(floor_io.ATLAS_CARDS) if a.all
             else ([a.card] if a.card else []))
    if not cards:
        ap.error("要么 --card，要么 --all")

    ok, failed = [], []
    for cid in cards:
        f, why = floor_io.resolve_atlas(cid)
        if f is None:
            # 「找不到卡」以前是 SystemExit，「缺 floor_inputs」以前是 SKIP ——
            # 两种处置不同（前者是打错字，后者是卡没填），这里保持区分。
            if why.startswith("找不到卡"):
                raise SystemExit(why)
            failed.append((cid, why))
            print("%-26s SKIP  %s" % (cid, why))
            continue
        # 打印用的是来源的原始 dict（analyze 的返回值）而不是 Floor 的字段：
        # --json 的输出形状必须与重构前逐字节一致，Floor 只是它的**视图**。
        r, spec = f.detail, f.inputs
        print("%-26s %+7.1f%%  %-10s floor=%.1fus actual=%sus  %-6s %s"
              % (cid, r["remaining"] * 100, r["gate"], r["t_floor_us"],
                 r["t_actual_us"], r["binding"], spec.get("t_source", "")))
        if a.write_card:
            wrote, msg = write_card(floor_io.ATLAS_CARDS / ("%s.yaml" % cid), r, spec)
            if wrote:
                ok.append(cid)
            else:
                failed.append((cid, msg))
                print("%-26s FAIL  %s" % (cid, msg))
        if a.json:
            print_floor(r, True)
    print("\n写卡 %d / 尝试 %d；未写 %d" % (len(ok), len(cards), len(failed)))
    for cid, why in failed:
        print("  - %s: %s" % (cid, why))
    return 0 if not failed else 1
