#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ablate.py — 消融归因（纯读：不跑 bench / 不写 runs/ / 不改卡片）。

attribution(m) = ms_ablated(m) - ms_champion（公式借 v7.1；实现不借它：
min(ms) 选冠军、无 MAD 噪声门、不认 measure_mode 三处都会把噪声变成结论）。
每个 tag 取 runs/loop 最新记录；显著性复用 engine/loop.py 的 _significant()
（= judge() 的 3×max(MAD)），可 claim 性复用 scripts/measure_mode.py ——
判据只有一份实现。spec 格式见 --selftest 的 _case。
用法: python3 -I scripts/ablate.py --card <卡> --spec /tmp/x.yaml [--json]
退出码: 0 ok；1 unexplained / no_significant_attribution；2 用法/spec/数据错误。
"""
import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / ".git").exists()),
                 Path(__file__).resolve().parents[1])
class SpecError(Exception):
    """spec/数据不可用 → 退出码 2（不是判据失败，判据失败是退出码 1）。"""
def _load(name, relpath):
    """按文件路径加载 —— python3 -I 不把仓库根加进 sys.path。"""
    try:
        s = importlib.util.spec_from_file_location(name, str(REPO_ROOT / relpath))
        m = importlib.util.module_from_spec(s)
        sys.modules[name] = m
        s.loader.exec_module(m)
        return m
    except Exception:                                            # noqa: BLE001
        return None

# 判据只有一份实现；加载失败 fail-closed（宁可不出结论，也不许长出第二套算法）。
_MM = _load("_ablate_measure_mode", "scripts/measure_mode.py")
_LOOP = _load("_ablate_loop", "Plugins/ops/engine/loop.py")
def _num(rec, key):
    v = rec.get(key)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
def read_records(card):
    """tag → 最新一条 run 记录；不取 min(ms)（那是「在噪声里选赢家」）。
    只读顶层 *.jsonl；noise/ 是 loop.py 的隔离区（未形成决策），不读。"""
    out, d = {}, card / "runs" / "loop"
    for f in sorted(d.glob("*.jsonl")) if d.is_dir() else []:
        for ln in f.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if isinstance(r, dict) and r.get("tag") and r.get("cand_ms") is not None:
                out[str(r["tag"])] = r
    return out
def load_spec(path):
    """读并校验 spec；缺项 → SpecError，不猜默认值。"""
    import yaml
    try:
        d = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as e:
        raise SpecError("读不到/解析不了 spec：%s" % e)
    if not all(isinstance(d.get(k), str) and d[k].strip() for k in ("baseline", "champion")):
        raise SpecError("spec 必须有 baseline 与 champion（要归因的 tag）")
    abl = d.get("ablations") or []
    ids = [str(it.get("id") or "") if isinstance(it, dict) else "" for it in abl]
    if not abl or not all(ids) or any(not str(it.get("tag") or "") for it in abl) \
            or len(set(ids)) != len(ids):
        raise SpecError("ablations 必须非空、id 唯一，且每项同时有 id 与 tag")
    return d
def _not_comparable(ref, rec, mm, loop):
    """ref 与 rec 的 ms 能否直接相减；返回不可比理由（空 = 可比）。每条封一个真错：
    measure_mode 不可 claim / 跨模式 —— 偏差逐实现变化，符号都能翻；
    shape 不同 —— 分母不同；base 漂移 / 缺 MAD —— 漂移会被算成「这项贡献」，
    MAD 缺失当 0 会让任意 delta 都显著（把没证据读成证据）。"""
    why, mode = [], str(rec.get("measure_mode") or "unknown")
    if not mm.claimable(mode):
        why.append("measure_mode=%s 不可 claim（只认 event_isolated/end_to_end）" % mode)
    ok, msg = mm.same_mode(ref.get("measure_mode"), mode)
    if not ok:
        why.append(msg)
    k = lambda r: json.dumps(r.get("shape"), sort_keys=True)
    if k(ref) != k(rec):
        why.append("shape 不同 —— 跨形状相减没有意义")
    if _num(ref, "mad") is None or _num(rec, "mad") is None:
        why.append("对照/变体缺 mad —— 3×max(MAD) 无从计算，不许当 0")
    b1, b2 = _num(ref, "base_ms"), _num(rec, "base_ms")
    if not b1 or not b2:
        why.append("缺 base_ms —— 无法确认两次测量对着同一个 baseline")
    else:
        verdict, msg = loop.judge(b1, _num(ref, "base_mad") or 0.0, b2, _num(rec, "base_mad") or 0.0)
        if verdict != "insignificant":
            why.append("baseline 漂移/噪声（%s）—— %s" % (verdict, msg))
    return why
def attribute(spec, records):
    """纯函数：spec + {tag: record} → report dict。不做 IO。"""
    mm, loop = _MM, _LOOP
    if mm is None or loop is None or not hasattr(loop, "_significant"):
        raise SpecError("加载不到 measure_mode.py 或 loop.py 的 _significant —— "
                        "判据必须复用单一真源，缺了就不出结论")
    missing = [t for t in [spec["baseline"], spec["champion"]]
               + [str(a["tag"]) for a in spec["ablations"]] if t not in records]
    if missing:
        raise SpecError("spec 引用的 tag 在 runs/loop/*.jsonl 里不存在：%s"
                        "（先用 loop.py run 测出来，不许在这里补测）" % ", ".join(missing))
    base, champ = records[spec["baseline"]], records[spec["champion"]]
    b_ms, c_ms, c_mad = _num(base, "cand_ms"), _num(champ, "cand_ms"), _num(champ, "mad")
    gain = b_ms - c_ms

    def sig(a_ms, a_mad, b_ms_, b_mad_):                   # 唯一判据入口
        return loop._significant({"median_ms": b_ms_, "mad_ms": b_mad_ or 0.0},
                                 {"median_ms": a_ms, "mad_ms": a_mad or 0.0})

    # 冠军必须相对基准显著更快，否则没有可归因的收益。
    cp = _not_comparable(base, champ, mm, loop)
    ok, _r, why = sig(c_ms, c_mad, b_ms, _num(base, "mad"))
    if not ok:
        cp.append("冠军相对基准不显著（%s）—— 没有可归因的收益" % why)
    elif c_ms >= b_ms:
        cp.append("冠军显著慢于基准（%s）—— 归因没有对象" % why)
    items = []
    for a in spec["ablations"]:
        rec, delta = records[str(a["tag"])], 0.0
        r_ms, r_mad = _num(rec, "cand_ms"), _num(rec, "mad")
        delta = r_ms - c_ms
        ok, _r, why = sig(r_ms, r_mad, c_ms, c_mad)
        bad = cp + _not_comparable(champ, rec, mm, loop)
        # 分母 = 冠军相对基准的总收益，不是「显著 delta 之和」：
        # 用字面公式时 share 恒为 1，[0.8,1.2] 校验永远通过 —— 判据自废。
        items.append({"id": str(a["id"]), "tag": str(a["tag"]), "ms": r_ms, "mad": r_mad,
                      "measure_mode": rec.get("measure_mode"), "delta_ms": delta,
                      "significant": bool(ok), "significant_why": why,
                      "claimable": not bad, "not_claimable_reason": "；".join(bad) or None,
                      "attribution_share": (delta / gain) if (ok and not bad and gain) else None})
    claimed = [it for it in items if it["attribution_share"] is not None]
    total = sum(it["attribution_share"] for it in claimed)
    if cp:
        status, reason = "no_significant_attribution", "冠军不可 claim：" + "；".join(cp)
    elif not claimed:
        status, reason = ("no_significant_attribution",
                          "没有可 claim 的显著项（不显著，或 measure_mode 不可 claim）")
    elif 0.8 <= total <= 1.2:
        status, reason = "ok", "显著项合计解释了总收益的 %.0f%%" % (total * 100)
    else:
        status, reason = "unexplained", ("显著项合计 share=%.2f 落在 [0.8, 1.2] 外：%s"
                                         % (total, "还有未归因的收益" if total < 0.8
                                            else "各项重叠/加总超过总收益（不满足可加性）"))
    side = lambda r: {"tag": r.get("tag"), "ms": _num(r, "cand_ms"), "mad": _num(r, "mad"),
                      "measure_mode": r.get("measure_mode")}
    return {"status": status, "status_reason": reason, "baseline": side(base),
            "champion": side(champ), "champion_problems": cp, "gain_ms": gain,
            "gain_pct": (gain / b_ms * 100.0) if b_ms else None,
            "items": items, "sum_share": total if claimed else None}
def _render(rep):
    f = lambda l, s: "%-4s %-14s %.5f ms  MAD=%s  %s" % (
        l, s["tag"], s["ms"], "-" if s["mad"] is None else "%.5f" % s["mad"], s["measure_mode"])
    out = ["卡片 %s  spec %s" % (rep["card"], rep["spec"]), f("基准", rep["baseline"]),
           f("冠军", rep["champion"]),
           "收益 %+.5f ms（%+.2f%%）" % (rep["gain_ms"], rep["gain_pct"] or 0.0),
           "%-16s %-14s %10s %8s  %s" % ("项", "tag", "delta_ms", "share", "判定")]
    for it in rep["items"]:
        # delta_ms > 0 = 去掉该项后更慢 = 该项有正贡献（消融的方向语义）
        v = ("not_claimable：" + it["not_claimable_reason"] if not it["claimable"]
             else ("显著（去掉后%s）" % ("变慢，该项有贡献" if it["delta_ms"] > 0 else "更快，该项负贡献"))
             if it["significant"] else "不显著（%s）" % it["significant_why"])
        out.append("%-16s %-14s %+10.5f %8s  %s" % (
            it["id"], it["tag"], it["delta_ms"], "-" if it["attribution_share"] is None
            else "%.3f" % it["attribution_share"], v))
    out += ["status %s —— %s" % (rep["status"], rep["status_reason"])]
    if rep["sum_share"] is not None:
        out += ["sum(share) %.3f" % rep["sum_share"]]
    return "\n".join(out)
def build_report(card, spec_path):
    rep = attribute(load_spec(spec_path), read_records(Path(card)))
    rep["card"], rep["spec"] = str(card), str(spec_path)
    return rep
def _r(tag, ms, mad=0.001, mode="event_isolated"):
    return {"tag": tag, "base_ms": 1.0, "base_mad": .001, "cand_ms": ms, "mad": mad,
            "measure_mode": mode, "shape": {"shape": "s"}}
def _case(recs, pairs):
    """合成 jsonl + spec 写成临时卡，走完整读取路径；返回 report 或 SpecError。"""
    import yaml
    with tempfile.TemporaryDirectory() as td:
        c = Path(td)
        (c / "runs/loop").mkdir(parents=True)
        (c / "runs/loop/x.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
        (c / "s.yaml").write_text(yaml.safe_dump({"baseline": "base", "champion": "champ",
            "ablations": [{"id": i, "tag": t} for i, t in pairs]}), encoding="utf-8")
        try:
            return build_report(c, c / "s.yaml")
        except SpecError as e:
            return e
def selftest():
    """每条判据都要证明「该报错时真会报错」，否则约束等于没写。"""
    def chk(label, recs, pairs, want, more=None):
        rep = _case(recs, pairs)
        if want == "SpecError":
            assert isinstance(rep, SpecError) and "不存在" in str(rep), (label, rep)
        else:
            assert rep["status"] == want, (label, rep)
        assert more is None or more(rep), (label, rep)
        print("[ok] " + label)
    b = _r("base", 1.0)
    chk("正向：两项各 0.5 → ok", [b, _r("champ", .9), _r("a", .95), _r("b", .95)],
        [("a", "a"), ("b", "b")], "ok", lambda r: abs(r["sum_share"] - 1) < 1e-9)
    chk("反向：全不显著 → no_significant_attribution（不硬凑数字）",
        [b, _r("champ", .9), _r("a", .9005), _r("b", .8995)],
        [("a", "a"), ("b", "b")], "no_significant_attribution",
        lambda r: r["sum_share"] is None)
    chk("反向：share < 0.8 → unexplained",
        [b, _r("champ", .9), _r("a", .92)], [("a", "a")], "unexplained",
        lambda r: r["sum_share"] < 0.8)
    chk("反向：share > 1.2（各项重叠）→ unexplained",
        [b, _r("champ", .9), _r("a", .98), _r("c", .98)],
        [("a", "a"), ("c", "c")], "unexplained", lambda r: r["sum_share"] > 1.2)
    chk("反向：back_to_back → not_claimable，不进合计",
        [b, _r("champ", .9), _r("a", .98, .001, "back_to_back")], [("a", "a")],
        "no_significant_attribution",
        lambda r: not r["items"][0]["claimable"] and r["items"][0]["attribution_share"] is None
        and "measure_mode" in r["items"][0]["not_claimable_reason"])
    chk("反向：spec 引用不存在的 tag → SpecError（退出码 2）",
        [b, _r("champ", .9), _r("a", .95)], [("a", "a"), ("x", "nope")], "SpecError")
    chk("反向：冠军不显著 → no_significant_attribution",
        [b, _r("champ", .999), _r("a", .95)], [("a", "a")],
        "no_significant_attribution", lambda r: bool(r["champion_problems"]))
    # 3×MAD 硬线：2.9× 不显著、3.5× 显著（判据来自 _significant，不是本脚本另算）
    r1 = _case([b, _r("champ", .9), _r("a", .9029)], [("a", "a")])
    r2 = _case([b, _r("champ", .9), _r("a", .9035)], [("a", "a")])
    assert not r1["items"][0]["significant"] and r2["items"][0]["significant"]
    print("[ok] 边界：2.9×MAD 不显著 / 3.5×MAD 显著")
    print("\nselftest PASS（正向 1 + 反向 6 + 边界 1）")
    return 0
def main(argv=None):
    ap = argparse.ArgumentParser(description="消融归因（纯读，判据复用 loop._significant）")
    ap.add_argument("--card", help="算子卡目录（读 <卡>/runs/loop/*.jsonl）")
    ap.add_argument("--spec", help="消融声明 YAML")
    ap.add_argument("--json", action="store_true", help="机器可读输出")
    ap.add_argument("--selftest", action="store_true", help="合成数据验证每条判据会报错")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.card or not a.spec:
        ap.error("--card 与 --spec 必填（或用 --selftest）")
    try:
        rep = build_report(Path(a.card).resolve(), Path(a.spec))
    except SpecError as e:
        print("ablate: %s" % e, file=sys.stderr)
        return 2
    print(json.dumps(rep, ensure_ascii=False, indent=2) if a.json else _render(rep))
    return 0 if rep["status"] == "ok" else 1

if __name__ == "__main__":
    sys.exit(main())
