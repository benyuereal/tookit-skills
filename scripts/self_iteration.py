#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""self_iteration —— 「自我迭代能力」评分器（六步链路是否接通）。

为什么需要它（与 scorecard 的 D 分是两件事）
--------------------------------------------
`scripts/scorecard.py` 的 D1–D5 衡量的是「卡上积累了多少**实测证据**」：
DS 均分、comparison 覆盖率、ledger 轮次数。它必须靠真跑实验才能涨。

但「证据数量」不等于「自我迭代能力」。自我迭代能力问的是：

    agent 能不能自己走完 定位 → 天花板 → 归因 → 选刀 → 实验 → 收敛 六步，
    不需要人干预？

两者必须分开算，否则「建了一堆工具但一张卡没跑」和「真跑通三张卡」
会在同一个分数上无法区分。

这是「能力」分，不是「数量」分
------------------------------
每一步的判据只问「这条链路有没有接通」（≥1 个实例满足），
不问「有多少张卡做到了」。后者作为**第二个数**单独报，叫 coverage：
六步全部接通的卡有几张 —— 那是「能力被真正用起来」的证据。

六步判据（每条都能机检，不许凭印象）
------------------------------------
  1 定位   op → 库.函数     atlas_query.py 的 where 有结果，且该卡 variants
                            至少一条带 import_path
  2 天花板  remaining+bound  该 skill 目录有 roofline.json，且含 bound +
                            remaining + t_actual_mode
  3 归因   bottleneck signal roofline.json 有 bottleneck_signal，**或**卡片
                            last_verdict.bottleneck.evidence[].reproduce
  4 选刀   top 候选           surfaces.yaml ≥10 条非 dead_end，且
                            dominant-to-surface.yaml 五个 dominant 全映射
  5 实验   真跑+判显著       runs/loop/*.jsonl ≥1 条 record，且 record 含
                            measure_mode 与 decision（unknown 不算声明）
  6 收敛   判停/换族         该卡 progress.stage ∈ {S6,S7}，且该卡 record
                            含 verdict / falsified 字段

注意 3 是 OR，不是 AND：有 roofline 文件 ≠ 有归因能力。文件里没有
bottleneck_signal 时，必须能在卡片上找到带 reproduce 路径的 bottleneck；
两处都没有才判 ✗。这正是最容易误判的一条（见 selftest）。

用法
----
    python3 -I scripts/self_iteration.py            # 人读
    python3 -I scripts/self_iteration.py --json
    python3 -I scripts/self_iteration.py --selftest
    python3 -I scripts/self_iteration.py --root <包根>   # 测试用

退出码：报告成功即 0；selftest 失败抛 AssertionError（非 0）。
"""
import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# 六步权重。和必须是 1.0（selftest 断言）。
WEIGHTS = {
    "locate": 0.15,
    "ceiling": 0.15,
    "attribute": 0.20,
    "pick": 0.15,
    "experiment": 0.20,
    "converge": 0.15,
}
STEPS = [
    ("locate", "1 定位"),
    ("ceiling", "2 天花板"),
    ("attribute", "3 归因"),
    ("pick", "4 选刀"),
    ("experiment", "5 实验"),
    ("converge", "6 收敛"),
]
DOMINANTS = ("hbm", "l2-miss", "lds-conflict", "vgpr-spill", "issue-stall")
MIN_SURFACES = 10
CONVERGED_STAGES = ("S6", "S7")
# measure_mode 的占位值。loop.py 在缺失时写入 "unknown"——它等于「没声明」，
# 不能因为字符串非空就放行。
UNSET_MODES = ("", "unknown", "none")


# ---------------------------------------------------------------- 读文件

def _load_yaml(path):
    try:
        import yaml
        return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except Exception:                                            # noqa: BLE001
        return {}


def _cards(root):
    d = root / "Plugins/ops/op-atlas/cards"
    if not d.is_dir():
        return {}
    return {p.stem: _load_yaml(p) for p in sorted(d.glob("*.yaml"))}


def _skill_dirs(root):
    d = root / "Plugins/ops/skills"
    if not d.is_dir():
        return []
    return [p for p in sorted(d.iterdir()) if p.is_dir()]


def _card_skill_dir(root, cid, card):
    """卡片 → 对应 skill 目录。优先 op_card 字段，否则 op-<card id>。"""
    oc = card.get("op_card") if isinstance(card, dict) else None
    if oc:
        p = root / str(oc)
        return p if p.is_dir() else p
    return root / "Plugins/ops/skills" / ("op-%s" % cid)


def _normalize_card(name):
    n = str(name or "").strip()
    if n.endswith(".yaml"):
        n = n[:-5]
    if n.startswith("op-"):
        n = n[3:]
    return n


def _roofline_ok(roof):
    """roofline.json 是否真的能算天花板：bound + remaining + t_actual_mode。"""
    if not isinstance(roof, dict):
        return False
    return (bool(roof.get("bound"))
            and roof.get("remaining") is not None
            and bool(roof.get("t_actual_mode")))


def _signal_ok(roof):
    """有 bottleneck_signal 字段且不是空壳。"""
    if not isinstance(roof, dict):
        return False
    return bool(roof.get("bottleneck_signal"))


def _bottleneck_evidence(card):
    """卡片 last_verdict.bottleneck.evidence[] 里带 reproduce 的条目。"""
    lv = (card or {}).get("last_verdict") or {}
    bn = lv.get("bottleneck") or {}
    ev = bn.get("evidence") if isinstance(bn, dict) else None
    if not isinstance(ev, list):
        return []
    return [e for e in ev
            if isinstance(e, dict) and str(e.get("reproduce") or "").strip()]


# ---------------------------------------------------------------- 1 定位

_atlas_cache = {}


def _atlas_module(root):
    """按路径加载 root 下的 atlas_query.py —— where 的唯一真源。

    用 importlib 按路径加载而不是 `import atlas_query`：模块自己的 REPO
    由 __file__ 推出，加载 root/scripts/atlas_query.py 后它读的就是
    root 下的 cards（测试临时仓库因此可用，不必把 PYTHONPATH 指过去）。
    """
    key = str(root)
    if key in _atlas_cache:
        return _atlas_cache[key]
    p = root / "scripts/atlas_query.py"
    mod = None
    if p.is_file():
        try:
            spec = importlib.util.spec_from_file_location(
                "_self_iter_atlas_%d" % (abs(hash(key)) % 10 ** 8), str(p))
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
        except Exception:                                        # noqa: BLE001
            mod = None
    _atlas_cache[key] = mod
    return mod


def _locate(root, cards, atlas):
    """返回 (capability, detail, hit_set)。

    hit_set = 被 atlas_query.where 认出、且 variants 里有 import_path 的卡。
    """
    import_paths = sum(1 for c in cards.values()
                       for v in (c.get("variants") or [])
                       if isinstance(v, dict) and str(v.get("import_path") or "").strip())
    if atlas is None:
        return False, "atlas_query.py 不存在，无法从 op 找到库.函数", set()
    try:
        bundled = atlas._cards()
    except Exception:                                            # noqa: BLE001
        return False, "atlas_query 无法读取卡片", set()
    hits = set()
    for cid, card in cards.items():
        if not any(isinstance(v, dict) and str(v.get("import_path") or "").strip()
                   for v in (card.get("variants") or [])):
            continue
        try:
            got = atlas.where(cid, cards=bundled)
        except Exception:                                        # noqa: BLE001
            got = []
        if got:
            hits.add(cid)
    passed = bool(hits) and import_paths > 0
    if passed:
        detail = "atlas_query 可用，%d 条 import_path" % import_paths
    elif import_paths == 0:
        detail = "atlas_query 可用，但没有任何 variants 带 import_path"
    else:
        detail = "atlas_query where 查不到带 import_path 的卡"
    return passed, detail, hits


# ---------------------------------------------------------------- 2/3 天花板/归因

def _rooflines(root, cards):
    """[(skill_dir, roofline_dict)]，按 skill 目录去重。"""
    seen = {}
    for d in _skill_dirs(root):
        p = d / "roofline.json"
        if p.is_file():
            try:
                seen[d] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:                                    # noqa: BLE001
                seen[d] = None
    # op_card 指向的 skill 可能不在 Plugins/ops/skills 下（当前布局不会，但别写死）
    for card in cards.values():
        d = _card_skill_dir(root, "", card)
        p = d / "roofline.json"
        if p.is_file() and d not in seen:
            try:
                seen[d] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:                                    # noqa: BLE001
                seen[d] = None
    return sorted(seen.items(), key=lambda kv: str(kv[0]))


def _ceiling(root, cards):
    roofs = _rooflines(root, cards)
    valid = [d for d, r in roofs if _roofline_ok(r)]
    passed = bool(valid)
    if passed:
        detail = "%d 个 skill 有 roofline.json（%d 个字段齐全）" % (
            len(roofs), len(valid))
    elif roofs:
        detail = "%d 个 skill 有 roofline.json，但没有一个同时含 bound + remaining + t_actual_mode" % len(roofs)
    else:
        detail = "0 个 skill 有 roofline.json"
    return passed, detail, {d for d, _ in roofs}, set(valid)


def _attribute(root, cards):
    roofs = _rooflines(root, cards)
    sig_skills = {d for d, r in roofs if _signal_ok(r)}
    ev_cards = {cid for cid, c in cards.items() if _bottleneck_evidence(c)}
    passed = bool(sig_skills or ev_cards)
    detail = ("%d 个 skill 有 bottleneck_signal；%d 张卡有 bottleneck.evidence[].reproduce"
              % (len(sig_skills), len(ev_cards)))
    if not passed:
        detail += "（两处都没有：有 roofline 文件 ≠ 有归因能力）"
    return passed, detail, sig_skills, ev_cards


# ---------------------------------------------------------------- 4 选刀

def _pick(root):
    ref = root / "Plugins/ops/skills/op-optimize/references"
    surfaces = _load_yaml(ref / "surfaces.yaml")
    if not isinstance(surfaces, list):
        surfaces = []
    rows = [s for s in surfaces if isinstance(s, dict)]
    active = [s for s in rows if str(s.get("verdict") or "") != "dead_end"]
    dead = [s for s in rows if str(s.get("verdict") or "") == "dead_end"]

    mapping = _load_yaml(ref / "dominant-to-surface.yaml")
    m = mapping.get("dominant_to_surface") if isinstance(mapping, dict) else {}
    m = m if isinstance(m, dict) else {}
    mapped = {k for k in DOMINANTS
              if isinstance(m.get(k), list) and m.get(k)}
    surfaces_ok = len(active) >= MIN_SURFACES
    mapping_ok = len(mapped) == len(DOMINANTS)
    passed = surfaces_ok and mapping_ok
    missing = [k for k in DOMINANTS if k not in mapped]
    detail = "%d 条 surface（%d 条 dead_end），%d 个 dominant %s" % (
        len(active), len(dead), len(mapped),
        "全映射" if mapping_ok else "缺 %s" % ",".join(missing))
    if not surfaces_ok:
        detail += "；非 dead_end 不足 %d 条" % MIN_SURFACES
    return passed, detail, active, mapping_ok


# ---------------------------------------------------------------- 5/6 实验/收敛

def _records(root, cards):
    """[(path, card_id, record)]。card_id 从 record.card 或路径推出。"""
    paths = []
    sd = root / "Plugins/ops/skills"
    if sd.is_dir():
        paths += sorted(sd.glob("*/runs/loop/*.jsonl"))
    rd = root / "runs/loop"
    if rd.is_dir():
        paths += sorted(rd.glob("*.jsonl"))
    # op_card 指定的 skill 可能在别处
    for card in cards.values():
        d = _card_skill_dir(root, "", card)
        if d.is_dir():
            paths += sorted((d / "runs/loop").glob("*.jsonl"))
    out, seen = [], set()
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except Exception:                                        # noqa: BLE001
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:                                    # noqa: BLE001, S112
                continue
            if not isinstance(rec, dict):
                continue
            cid = _normalize_card(rec.get("card"))
            if not cid:
                parts = p.parts
                if "skills" in parts:
                    i = parts.index("skills")
                    if i + 1 < len(parts):
                        cid = _normalize_card(parts[i + 1])
            out.append((p, cid, rec))
    return out


def _record_measured(rec):
    """record 是否算「真跑了并判显著」：有 decision，且 measure_mode 真声明了。"""
    mode = str(rec.get("measure_mode") or "").strip().lower()
    return bool(str(rec.get("decision") or "").strip()) and mode not in UNSET_MODES


def _record_converged(rec):
    """record 是否带收敛判决字段（verdict 或 falsified，键存在即算）。"""
    return ("verdict" in rec) or ("falsified" in rec)


def _experiment(root, cards, recs):
    measured = [(p, cid, r) for p, cid, r in recs if _record_measured(r)]
    passed = bool(measured)
    detail = "%d 条 record，%d 条含 measure_mode 与 decision" % (len(recs), len(measured))
    if recs and not measured:
        detail += "（record 缺 measure_mode 或 decision）"
    return passed, detail, {cid for _, cid, _ in measured}


def _converge(root, cards, recs):
    converged = {cid for cid, c in cards.items()
                 if str(((c.get("progress") or {}).get("stage")) or "").upper()
                 in CONVERGED_STAGES}
    with_v = {cid for _, cid, r in recs if _record_converged(r)}
    both = converged & with_v
    passed = bool(both)
    detail = "%d 张卡到 S6/S7，%d 张卡有 verdict/falsified record" % (
        len(converged), len(with_v))
    return passed, detail, converged, with_v


# ---------------------------------------------------------------- 汇总

def collect(root=None):
    root = Path(root) if root else REPO
    cards = _cards(root)
    atlas = _atlas_module(root)
    recs = _records(root, cards)

    locate_ok, locate_d, hits = _locate(root, cards, atlas)
    ceil_ok, ceil_d, roofs, valid_roofs = _ceiling(root, cards)
    attr_ok, attr_d, sig_skills, ev_cards = _attribute(root, cards)
    pick_ok, pick_d, _, _ = _pick(root)
    exp_ok, exp_d, measured_cards = _experiment(root, cards, recs)
    conv_ok, conv_d, converged, verdict_cards = _converge(root, cards, recs)

    steps = {
        "locate": {"label": "1 定位", "weight": WEIGHTS["locate"],
                   "passed": locate_ok, "detail": locate_d},
        "ceiling": {"label": "2 天花板", "weight": WEIGHTS["ceiling"],
                    "passed": ceil_ok, "detail": ceil_d},
        "attribute": {"label": "3 归因", "weight": WEIGHTS["attribute"],
                      "passed": attr_ok, "detail": attr_d},
        "pick": {"label": "4 选刀", "weight": WEIGHTS["pick"],
                 "passed": pick_ok, "detail": pick_d},
        "experiment": {"label": "5 实验", "weight": WEIGHTS["experiment"],
                       "passed": exp_ok, "detail": exp_d},
        "converge": {"label": "6 收敛", "weight": WEIGHTS["converge"],
                     "passed": conv_ok, "detail": conv_d},
    }
    score = round(10 * sum(s["weight"] for s in steps.values() if s["passed"]), 1)

    per_card = {}
    for cid, card in cards.items():
        skill = _card_skill_dir(root, cid, card)
        roof = {}
        rp = skill / "roofline.json"
        if rp.is_file():
            try:
                roof = json.loads(rp.read_text(encoding="utf-8"))
            except Exception:                                    # noqa: BLE001
                roof = {}
        card_recs = [r for _, rc, r in recs if rc == cid]
        c_locate = cid in hits
        c_ceiling = _roofline_ok(roof)
        c_attr = _signal_ok(roof) or bool(_bottleneck_evidence(card))
        c_exp = any(_record_measured(r) for r in card_recs)
        stage_ok = (str(((card.get("progress") or {}).get("stage")) or "").upper()
                    in CONVERGED_STAGES)
        c_conv = stage_ok and any(_record_converged(r) for r in card_recs)
        # 第 4 步是仓库级能力：选刀表不接通的仓库，单卡也不算全通。
        flags = {"locate": c_locate, "ceiling": c_ceiling, "attribute": c_attr,
                 "pick": pick_ok, "experiment": c_exp, "converge": c_conv}
        per_card[cid] = {"steps": flags, "all": all(flags.values())}

    full = sorted(cid for cid, v in per_card.items() if v["all"])
    return {
        "root": str(root),
        "score": score,
        "max": 10,
        "weights": dict(WEIGHTS),
        "weights_sum": round(sum(WEIGHTS.values()), 12),
        "steps": steps,
        "coverage": {"all_six": len(full), "cards": full},
        "cards": per_card,
    }


def render(result):
    lines = ["自我迭代能力     %.1f / 10" % result["score"]]
    for key, _ in STEPS:
        s = result["steps"][key]
        mark = "✓" if s["passed"] else "✗"
        lines.append("  [%s] %-6s %s" % (mark, s["label"], s["detail"]))
    lines.append("")
    lines.append("六步全通的卡   %d 张" % result["coverage"]["all_six"])
    return "\n".join(lines)


# ---------------------------------------------------------------- selftest

def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_fixture(root, drop=None):
    """造一个「六步全通」的临时仓库；drop 指定要故意漏掉的字段/文件。"""
    root = Path(root)
    # 1 定位需要真的 atlas_query.py —— 拷仓库里的唯一真源
    src = Path(__file__).resolve().parent / "atlas_query.py"
    if src.is_file():
        _write(root / "scripts/atlas_query.py", src.read_text(encoding="utf-8"))

    card = """\
id: foo
op_card: Plugins/ops/skills/op-foo
variants:
- repo: lib
  import_path: lib.foo
last_verdict:
  bottleneck:
    bound: memory
    evidence:
    - kind: ablation
      reproduce: Plugins/ops/skills/op-foo/ledger.md#轮 1
progress:
  stage: S7
"""
    if drop == "card_import_path":
        card = card.replace("  import_path: lib.foo\n", "")
    if drop in ("card_bottleneck", "attribute"):
        card = card.replace("  bottleneck:\n    bound: memory\n", "")
        card = card.replace("    evidence:\n    - kind: ablation\n"
                            "      reproduce: Plugins/ops/skills/op-foo/ledger.md#轮 1\n", "")
    if drop == "stage":
        card = card.replace("  stage: S7", "  stage: S4")
    _write(root / "Plugins/ops/op-atlas/cards/foo.yaml", card)

    roof = {"bound": "memory", "remaining": 0.1088,
            "t_actual_mode": "event_isolated"}
    if drop not in ("no_signal", "attribute"):
        roof["bottleneck_signal"] = {"bound": "memory", "kind": "ablation"}
    if drop == "remaining":
        roof.pop("remaining")
    if drop != "roofline":
        _write(root / "Plugins/ops/skills/op-foo/roofline.json",
               json.dumps(roof, ensure_ascii=False))
    # drop == "no_signal"：roofline 照写，但去掉 bottleneck_signal；
    # 归因只能靠卡片的 evidence。用于验证「文件在 ≠ 能力在」。

    rec = {"card": "op-foo", "decision": "ACCEPTED_BITEXACT",
           "measure_mode": "event_isolated", "falsified": False}
    if drop == "measure_mode":
        rec.pop("measure_mode", None)
    if drop == "falsified":
        rec.pop("falsified", None)
    _write(root / "Plugins/ops/skills/op-foo/runs/loop/20260101.jsonl",
           json.dumps(rec, ensure_ascii=False) + "\n")

    ref = root / "Plugins/ops/skills/op-optimize/references"
    rows = ["- id: s%d\n  statement: s\n  verdict: active\n" % i
            for i in range(1, MIN_SURFACES + 1)]
    if drop == "all_dead_end":
        rows = ["- id: s%d\n  statement: s\n  verdict: dead_end\n" % i
                for i in range(1, MIN_SURFACES + 4)]
    _write(ref / "surfaces.yaml", "".join(rows))
    mapping = {k: ["s1"] for k in DOMINANTS}
    if drop == "dominant":
        mapping.pop("vgpr-spill", None)
    _write(ref / "dominant-to-surface.yaml",
           "dominant_to_surface:\n"
           + "".join("  %s:\n  - s1\n" % k for k in mapping))


def selftest():
    assert round(sum(WEIGHTS.values()), 12) == 1.0, "六步权重和必须为 1.0"

    with tempfile.TemporaryDirectory() as td:
        empty = Path(td) / "empty"
        empty.mkdir()
        r = collect(empty)
        assert r["score"] == 0.0, "空仓库必须 0 分，实得 %s" % r["score"]
        assert r["coverage"]["all_six"] == 0
    print("[ok] 空仓库 → 0 分，不崩")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "full"
        _make_fixture(root)
        r = collect(root)
        assert r["score"] == 10.0, "六步全通的仓库必须满分，实得 %s: %s" % (
            r["score"], {k: v["passed"] for k, v in r["steps"].items()})
        assert all(s["passed"] for s in r["steps"].values())
        assert r["coverage"]["all_six"] == 1, "应算出 1 张六步全通的卡"
    print("[ok] 构造的六步全通仓库 → 10.0，coverage=1")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "nosig"
        _make_fixture(root, drop="no_signal")
        r = collect(root)
        assert r["steps"]["attribute"]["passed"], "卡片有 evidence.reproduce 时第 3 步应通"
        # 再把卡片 evidence 也删掉、且 roofline 里也没有 signal：两处都没有，才判 ✗
        _make_fixture(root, drop="card_bottleneck")
        (root / "Plugins/ops/skills/op-foo/roofline.json").unlink()
        r2 = collect(root)
        assert not r2["steps"]["attribute"]["passed"], \
            "roofline 无 bottleneck_signal 且卡片无 evidence 时第 3 步不得分"
    print("[ok] 有 roofline 文件但无 bottleneck_signal / 无 evidence → 第 3 步 ✗")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "dead"
        _make_fixture(root, drop="all_dead_end")
        r = collect(root)
        assert not r["steps"]["pick"]["passed"], "全是 dead_end 时第 4 步不得分"
        assert r["score"] == 8.5, "只该扣 0.15×10，实得 %s" % r["score"]
    print("[ok] surfaces 全 dead_end → 第 4 步 ✗")

    for drop, step, delta in (("measure_mode", "experiment", 2.0),
                              ("card_import_path", "locate", 1.5),
                              ("remaining", "ceiling", 1.5),
                              ("attribute", "attribute", 2.0),                              ("stage", "converge", 1.5),
                              ("falsified", "converge", 1.5),
                              ("dominant", "pick", 1.5)):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "drop"
            _make_fixture(root, drop=drop)
            r = collect(root)
            assert not r["steps"][step]["passed"], \
                "故意漏 %s，第 %s 步必须判 ✗（不许静默给分）" % (drop, step)
            assert abs(r["score"] - (10.0 - delta)) < 1e-9, \
                "漏 %s 应扣 %.1f，实得 %s" % (drop, delta, r["score"])
            assert r["coverage"]["all_six"] == 0
    print("[ok] 反向：逐个漏字段，对应步骤变 ✗ 且扣分精确")

    print("\nselftest PASS")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="自我迭代能力六步评分器")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--root", help="包根（默认本脚本的上一级）")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    result = collect(a.root)
    if a.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(render(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
