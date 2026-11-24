#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ingest.py —— 按 Plugins/ops/ingest/schema.yaml 从外部语料抽吸纳记录（jsonl）。

只抽机器能抽的：RESULTS*.md 的已知表格/散文、kernel 里的 #define。
**不写人的判断**：属实、是否适用、吸不吸收，一律不进机器字段；抽不到写 missing
并给原因，绝不补 0，也不拿文档的数字冒充我方复跑。
"""
import argparse, json, re, sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "mhc-ingest/1"
HUMAN = ("our_measured_value", "verdict_note", "our_significance", "mechanism",
         "preconditions", "architecture_applicable", "absorb", "absorb_target",
         "absorb_action")
REQ = {k: ("metric", "claim_text", "claimed_value", "claim_unit",
           "baseline_desc", "claim_verdict") for k in ("claim", "measurement")}
REQ.update({k: ("metric", "claim_text", "claimed_value", "claim_unit")
            for k in ("define", "caliber")})
REQ["manifest"] = ()
NUM = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def cells(line):
    """按未转义竖线切：`\\|` 是内容（RESULTS2 的 `layer \\|\\| CE`），不处理会切错列。"""
    ps = re.split(r"(?<!\\)\|", line.strip())
    if ps and not ps[0].strip():
        ps = ps[1:]
    if ps and not ps[-1].strip():
        ps = ps[:-1]
    return [c.strip() for c in ps]


def clean(s):
    return re.sub(r"[`*]", "", s or "").replace("\\|", "|").strip()


def num(s):
    m = NUM.search(clean(s))
    return float(m.group()) if m else None


def nums(s):
    # 区间 208-251 的短横是分隔符不是负号；e-6 前是字母，不受影响
    return [float(x) for x in NUM.findall(re.sub(r"(?<=\d)-(?=\d)", " ", clean(s)))]


def is_sep(line):
    """分隔行 |---|---|。数据行也有短横（1456–1827），不能整行匹配长短横。"""
    c = cells(line)
    return bool(c) and all(re.fullmatch(r":?-{3,}:?", x) for x in c)


def delta(claimed, ours):
    """派生量。缺一侧 → None；绝不用 0 表示一致，否则没跑与跑对不可分。"""
    if ours is None or not isinstance(claimed, (int, float)) or claimed == 0:
        return None
    return round((ours - claimed) / claimed * 100.0, 2)


def mk(f, i, line, kind, stage, metric, claimed=None, unit=None, **kw):
    """构造一条记录。claimed=None 进 missing，不 pad 默认值。"""
    sid = "%s-L%d-%s%s" % (Path(f).stem, i, re.sub(
        r"[^0-9A-Za-z_.-]+", "-", clean(stage)).strip("-")[:48], kw.pop("suffix", ""))
    d = {"schema": SCHEMA, "id": sid, "kind": kind, "stage": stage,
         "metric": metric, "claim_text": line.strip(), "claim_unit": unit,
         "provenance": {"file": f, "line": i, "quote": line.rstrip()}}
    if claimed is not None:
        d["claimed_value"] = claimed
    for k in ("shape", "baseline_desc", "baseline_value", "denominator_gbs",
              "falsified", "has_control", "doc_measured_value"):
        if kw.get(k) is not None:
            d[k] = kw[k]
    if metric == "speedup_x" and isinstance(claimed, (int, float)):
        d["significance"] = ("significant" if claimed >= 1.03 else
                             "marginal" if claimed >= 1.005 else "none")
    # 无我方复跑，只能是不复现；定义/口径不是断言
    d["claim_verdict"] = ("not_applicable" if kind in ("define", "caliber", "manifest")
                          else "not_reverified")
    need = list(REQ.get(kind, ())) + (["denominator_gbs"]
                                      if metric in ("bandwidth_gbs", "range_gbs") else [])
    for k in need:
        if d.get(k) in (None, ""):
            d.setdefault("missing", {})[k] = "抽不到：%s" % (kw.get("why") or "该行缺少此列")
    d["human_pending"] = [h for h in HUMAN if h not in d]
    return d


# ── 表格处理器：(file, line, lineno, ctx) -> [record...]，认行不认文件 ──────────
def h_summary(f, ln, i, ctx):
    c = cells(ln)
    if len(c) < 4:
        return []
    m = re.search(r"m=(\d+)", clean(c[0]))
    sh = "m=%s" % m.group(1) if m else None
    # doc_measured_value = 文档 new 列实测时间；不抄 claim 倍率（那是自证）
    out = [mk(f, i, ln, "claim", "total" if sh else clean(c[0]), "speedup_x",
              num(c[3]), "x", shape=sh, baseline_desc=ctx["b"],
              baseline_value=num(c[1]), doc_measured_value=num(c[2]))]
    if len(c) >= 5 and num(c[4]) is not None:
        out.append(mk(f, i, ln, "measurement", "proj", "bandwidth_gbs", num(c[4]),
                      "gbs", shape=sh, baseline_desc="peak %s" % ctx["d"],
                      denominator_gbs=ctx["d"], suffix="-bw"))
    return out


def h_kernel(f, ln, i, ctx):
    c = cells(ln)
    if len(c) < 3 or "—" in c[2]:
        return []
    us, v = "us" in c[2], nums(c[2])
    return [mk(f, i, ln, "measurement", clean(c[0]).split("(")[0].strip(),
               "latency_us" if us else "range_gbs", v[0] if len(v) == 1 else v,
               "us" if us else "gbs", baseline_desc="peak %s" % ctx["d"],
               denominator_gbs=None if us else ctx["d"])]


def h_fusion(f, ln, i, ctx):
    c = cells(ln)
    if len(c) < 4:
        return []
    v = nums(c[1])
    return [mk(f, i, ln, "measurement", clean(c[0]), "latency_us",
               v[0] if v else None, "us", baseline_desc="unfused (%s)" % clean(c[3]),
               falsified="slower" in c[3], has_control=True, why="无 m=4096 绝对值")]


def h_pattern(f, ln, i, ctx):
    b = [float(x) for x in re.findall(r"\*\*([\d.]+)\*\*", ln)]
    return [mk(f, i, ln, "measurement", "proj_pat_ceiling", "range_gbs", b, "gbs",
               shape=clean(cells(ln)[0]), baseline_desc="linear %s" % ctx["d"],
               denominator_gbs=ctx["d"])] if b else []


def h_sens(f, ln, i, ctx):
    c = cells(ln)
    if len(c) < 4 or num(c[3]) is None or "x" not in c[3]:
        return []
    return [mk(f, i, ln, "claim", "sensitivity", "speedup_x", num(c[3]), "x",
               shape="m=16384", baseline_desc="sensitivity §9",
               falsified="slower" in (c[3] + c[-1]).lower())]


def h_verdict2(f, ln, i, ctx):
    c = cells(ln)
    if len(c) < 3:
        return []
    h = clean(c[2])
    fals = bool(re.search(r"falsified|rejected|not viable|slower", c[1] + h, re.I))
    if "GB/s" in h:
        metric, unit, v = "bandwidth_gbs", "gbs", num(h)
    elif "%" in h or "x" in h:
        b = re.search(r"\*\*([^*]*%[^*]*)\*\*" if "%" in h else r"\*\*([^*]*x[^*]*)\*\*", h)
        v = nums(b.group(1) if b else h)
        metric, unit = ("pct", "pct") if "%" in h else ("speedup_x", "x")
        v = v[0] if len(v) == 1 else v
    else:
        metric, unit, v = "define_value", "string", h
    return [mk(f, i, ln, "claim", clean(c[0]), metric, v, unit,
               baseline_desc="RESULTS2 §0 自述对照", falsified=fals, has_control=False,
               denominator_gbs=ctx["d"] if unit == "gbs" else None)]


def h_w22(f, ln, i, ctx):
    c = cells(ln)
    if len(c) < 3 or num(c[2]) is None:
        return []
    n = clean(c[0])
    return [mk(f, i, ln, "measurement", n, "bandwidth_gbs", num(c[2]), "gbs",
               baseline_desc="MB=1 control (%s)" % clean(c[3]),
               denominator_gbs=ctx["d"], has_control=True,
               falsified=bool(re.search(r"MB=[234]", n)))]


def h_w24(f, ln, i, ctx):
    c = cells(ln)
    return [mk(f, i, ln, "measurement", "proj4_loop_shape", "speedup_x", num(c[3]),
               "x", shape="m=%s" % clean(c[0]), baseline_desc="proj3 S=1",
               baseline_value=num(c[1]))] if len(c) >= 4 and num(c[3]) is not None else []


def h_route(f, ln, i, ctx):
    c = cells(ln)
    if len(c) < 2:
        return []
    v = nums(c[1])
    return [mk(f, i, ln, "claim", "tried:" + clean(c[0]), "outcome",
               v[0] if len(v) == 1 else (v or clean(c[1])),
               "gbs" if "GB/s" in c[1] else "string", baseline_desc="上一版实现（§5）",
               denominator_gbs=ctx["d"] if "GB/s" in c[1] else None,
               falsified=bool(re.search(r"slower|worse|rejected|does not fit|no",
                                        c[1], re.I)))]


def h_vgpr(f, ln, i, ctx):
    c = cells(ln)
    return [mk(f, i, ln, "measurement", clean(c[0]) + ":vgpr", "count", int(num(c[1])),
               "count", baseline_desc="aicc resource report")
            ] if len(c) >= 2 and num(c[1]) is not None else []


def h_overlap(f, ln, i, ctx):
    c = [x for x in cells(ln) if x]   # 表头有 `layer || CE`，空串按内容剔除
    if len(c) < 4:
        return []
    hid = next((x for x in c if re.search(r"\d\s*%", clean(x)) and "of" in clean(x)), None)
    return [mk(f, i, ln, "measurement", "overlap:" + clean(c[0]), "pct",
               num(hid) if hid else None, "pct", baseline_desc="sequential 对照",
               has_control=True, why="无 'N % of' 隐藏比例")]


def h_correctness(f, ln, i, ctx):
    c = cells(ln)
    v = nums(c[2]) if len(c) >= 3 and "/" in c[2] else []
    return [mk(f, i, ln, "measurement", "correctness:" + clean(c[0]), "abs_error",
               v[0], "string", shape="m=%s" % clean(c[1]), baseline_desc="fp64 gold")
            ] if v else []


TABLES = [
    (r"\|\s*baseline\s*\|\s*new\s*\|", h_summary), (r"traffic \(bytes moved\)", h_kernel),
    (r"vs unfused", h_fusion), (r"\|\s*run per row\s*\|", h_pattern),
    (r"\|\s*proj bandwidth\s*\|", h_sens), (r"\|\s*direction\s*\|\s*result", h_verdict2),
    (r"\|\s*variant\s*\|\s*best split", h_w22), (r"proj3 \(mhc\.so\), S=1", h_w24),
    (r"\|\s*route\s*\|\s*result\s*\|", h_route), (r"\|\s*kernel\s*\|\s*VGPRs", h_vgpr),
    (r"\|\s*schedule\s*\|\s*wall", h_overlap), (r"vs fp64 gold", h_correctness),
]


def prose(f, i, ln, ctx):
    """散文只认分母声明与天花板断言；其余不抽，宁缺毋滥。"""
    out = []
    if "denominator" in ln and "GB/s" in ln and "1870" in ln and num(ln):
        out.append(mk(f, i, ln, "caliber", "bandwidth_denominator", "bandwidth_gbs",
                      num(ln), "gbs", baseline_desc="流式读上限",
                      denominator_gbs=num(ln)))
    m = re.search(r"ceiling is ([\d.]+)x", ln)
    if m:
        out.append(mk(f, i, ln, "claim", "ceiling", "speedup_x", float(m.group(1)),
                      "x", baseline_desc="文档自述上限"))
    return out


def find(root, pat):
    for p in sorted(root.glob("*.md")):
        for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
            if re.search(pat, ln):
                return ln


def defs(root):
    out = []
    for p in sorted(root.glob("kernels/*.hip")) + sorted(root.glob("explore/proj4.hip")):
        rel = str(p.relative_to(root))
        for i, ln in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            m = re.match(r"\s*#define\s+(\w+)\s+(\d+)\s*$", ln)   # 只认字面量
            if m:
                out.append(mk(rel, i, ln, "define", "define:" + m.group(1),
                              "define_value", int(m.group(2)), "count"))
    return out


def extract(root):
    root = Path(root)
    dl, bl = find(root, r"(denominator.*GB/s|GB/s.*denominator)"), find(root, r"^\s*Baseline\s*=")
    ctx = {"d": nums(dl)[0] if dl else None, "b": clean(bl) if bl else None}
    recs = []
    for fn in ("RESULTS.md", "RESULTS2.md"):
        if not (root / fn).is_file():
            continue
        h = None
        for i, ln in enumerate((root / fn).read_text(encoding="utf-8",
                                                     errors="replace").splitlines(), 1):
            if ln.startswith("|"):
                if h is None:                  # 首行是表头：认 handler，不当数据
                    h = next((x for p, x in TABLES if re.search(p, ln)), False)
                    continue
                if h and not is_sep(ln):       # 次行是分隔行，跳过
                    recs += h(fn, ln, i, ctx)
            else:
                h = None
                recs += prose(fn, i, ln, ctx)
    return recs + defs(root)


def build(root, out):
    root, recs = Path(root), None
    recs = extract(root)
    counts = {}
    for r in recs:
        counts[r["kind"]] = counts.get(r["kind"], 0) + 1
    rd = root / "README.md"
    man = {"schema": SCHEMA, "kind": "manifest", "id": "manifest", "stage": "manifest",
           "record_format": "jsonl; 一行一条",
           "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "source_root": str(root), "counts": counts,
           "schema_yaml": "Plugins/ops/ingest/schema.yaml",
           "provenance": {"file": "README.md", "line": 1,
                          "quote": (rd.read_text(encoding="utf-8").splitlines()
                                    or [""])[0] if rd.is_file() else ""}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(x, ensure_ascii=False)
                             for x in [man] + recs) + "\n", encoding="utf-8")
    return recs, out


def selftest():
    """正向证明该抽真抽到；反向证明不该抽真报 missing。"""
    import tempfile
    d = Path(tempfile.mkdtemp())
    (d / "kernels").mkdir()
    (d / "README.md").write_text("# s\n", encoding="utf-8")
    (d / "kernels/k.hip").write_text("#define A 4096\n#define B (A * 2)\n", encoding="utf-8")
    (d / "RESULTS.md").write_text("\n".join([
        "All bandwidth use **1870 GB/s** as the denominator.", "Baseline = `ref`.", "",
        "| | baseline | new | | proj |", "|---|---|---|---|---|",
        "| m=4096 total | 1000 us | 500 us | **2.00x** | 500 GB/s |",
        "| m=8192 total | 1000 us | 500 us | ??? | n/a |", "", "the ceiling is 2.06x"]),
        encoding="utf-8")
    recs = extract(d)
    by = {(r["stage"], r.get("shape")): r for r in recs}
    g = by[("total", "m=4096")]
    assert g["claimed_value"] == 2.0 and g["baseline_value"] == 1000.0, g
    assert g["doc_measured_value"] == 500.0 and g["claim_verdict"] == "not_reverified"
    assert "architecture_applicable" in g["human_pending"]
    assert by[("proj", "m=4096")]["claimed_value"] == 500.0
    assert by[("proj", "m=4096")]["denominator_gbs"] == 1870.0
    assert by[("ceiling", None)]["claimed_value"] == 2.06
    print("[ok] 正向：表格行/带宽列/散文天花板的数字真抽到")
    assert "claimed_value" in by[("total", "m=8192")]["missing"]
    assert "claimed_value" not in g.get("missing", {})
    print("[ok] 反向：行内是 ??? → missing 且带原因，未补 0")
    assert set(g["human_pending"]) == set(HUMAN) and not any(h in g for h in HUMAN)
    print("[ok] 反向：human 字段全在 human_pending，机器一个没填")
    assert not [r for r in recs if r["stage"] == "define:B"]
    assert [r for r in recs if r["stage"] == "define:A"]
    print("[ok] 反向：宏表达式未被当数值定义")
    assert delta(2.0, 1.9) == -5.0 and delta(2.0, None) is None
    print("[ok] delta_pct 依赖我方复跑；缺失时为 None 而非 0")
    for r in recs:
        p = r["provenance"]
        assert p["quote"] in (d / p["file"]).read_text(
            encoding="utf-8").splitlines()[p["line"] - 1]
    print("[ok] provenance：每条 quote 都能在原文对应行找到")
    print("selftest PASS")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", nargs="?", default="/workspace/mhc_kernels")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] /
                                         "Plugins/ops/ingest/mhc/records.jsonl"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    recs, out = build(a.root, Path(a.out))
    print("抽出 %d 条 → %s（missing %d 条，human_pending %d 处）"
          % (len(recs), out, sum(1 for r in recs if r.get("missing")),
             sum(len(r.get("human_pending", [])) for r in recs)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
