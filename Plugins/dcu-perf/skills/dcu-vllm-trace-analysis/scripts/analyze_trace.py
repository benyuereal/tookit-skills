#!/usr/bin/env python3
"""Analyze a vLLM V1 torch.profiler trace (.pt.trace.json[.gz]).

Two-step method (as designed on 2026-08-05 with the sg-t1 qwen3.5 w8a16 trace):

Step 1 - locate profiling stages. vLLM V1 names stages like
    execute_context_1(76)_generation_0(0)      <- prefill (ctx_len=76, gen_len=0)
    execute_context_0(0)_generation_1(1)       <- decode  (ctx_len=0,  gen_len=1)
  Each stage name appears as an event on the CPU executor thread AND on the
  GPU side (device pid/tid). Stage regex + CPU/GPU pairing -> spans.

Step 2 - xlsx per-category kernel stats. Classify each kernel by name using
  precedence rules (see CATEGORY_RULES); aggregate per phase (prefill/decode)
  and per single stage:
    total_ms / per-kernel avg_ms / pct of kernel sum / pct of stage dur /
    CPU launch overhead (sum of matching hipModuleLaunchKernel dur) / GPU bubble.

Usage:
    python analyze_trace.py TRACE [--out OUT.xlsx] [--categories cats.json]
    python analyze_trace.py TRACE --text        # no pandas needed, print tables
"""
import gzip
import json
import re
import sys
from collections import Counter

# ---------------------------------------------------------------------------
# Category rules. Ordered list of (category_name, regex) — FIRST match wins.
# Precedence matters: e.g. MOE marlin must beat w8a16; w8a16 must beat plain gemm.
# ---------------------------------------------------------------------------
DEFAULT_CATEGORY_RULES = [
    ("triton_poi",   r"triton_poi"),
    ("triton_red",   r"triton_red|triton_per"),   # torch.compile reduction / persist-reduction fusions
    ("moe_prep",     r"moe_align|align_block|moe_sum|silu|topk|moe_.*(reindex|sum|align)"),
    ("moe_gemm",     r"MOE_|MARLIN|moe_gemm|fused_moe|moe_c::|moe\.gemm|expert"),
    ("attn",         r"flash|attention|mla|fla|kkt|fused_h_o|gated_delta|gdn|causal_conv|recurrent|chunk_gated|mamba|_attn"),
    ("w8a16_gemm",   r"gemm_w8a16|w8a16|deepgemm"),
    ("gemm",         r"^Cij|rocblas|gemm|GEMM"),
    ("elementwise",  r"rmsnorm|layernorm|mrope|elementwise|pointwise|residual|gelu|_pw_|_fwd.*kernel"),
    ("memory",       r"memcpy|memset|zero_kv|fill|_copy_|direct_copy|Memset|Memcpy"),
    ("communication", r"nccl|allreduce|all_reduce|all_gather|reduce_scatter|send|recv|barrier"),
    ("other",        r".*"),
]

STAGE_RE = re.compile(r"execute_(?:context|generation)_\d+\(\d+\)_generation_\d+\(\d+\)")
LAUNCH_NAME_RE = re.compile(r"hipModuleLaunchKernel|cudaLaunchKernel|hipLaunchKernel")

# -- kernel-name shortening (demangle + strip C++ signature) --
_HAS_CPPFILT = None
_demangle_cache = {}


def _demangle(name):
    """Demangle an Itanium _Z... name to its unqualified function name."""
    global _HAS_CPPFILT
    if name in _demangle_cache:
        return _demangle_cache[name]
    if _HAS_CPPFILT is None:
        import shutil
        _HAS_CPPFILT = shutil.which("c++filt") is not None
    out = name
    if _HAS_CPPFILT:
        try:
            import subprocess
            r = subprocess.run(["c++filt", "-n", name], capture_output=True, text=True, timeout=3)
            if r.returncode == 0 and r.stdout.strip():
                out = r.stdout.strip()
        except Exception:
            pass
    else:
        # fallback: last "<len>name" token right before template args (I) / end of qual-name (E)
        toks = re.findall(r"(\d+[A-Za-z_]\w*)(?=[IE])", name)
        if toks:
            out = toks[-1]
    _demangle_cache[name] = out
    return out


def short_name(name):
    """Collapse a kernel name to something readable: strip 'void', demangle _Z...,
    cut the C++ signature at '<'/'(' and the namespace, keep the function name."""
    n = name.strip()
    if n.startswith("void"):
        n = n[4:].strip()
    if n.startswith("_Z"):
        n = _demangle(n)
    n = n.replace("(anonymous namespace)", "")
    # cut template args and the trailing C++ argument list
    for sep in ("<", "("):
        i = n.find(sep)
        if i > 0:
            n = n[:i]
    parts = [p for p in n.split("::") if p]
    return parts[-1] if parts else n


def load_trace(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)


def detect_device_thread(events):
    """Device tid/pid = the (pid,tid) hosting the most kernel events."""
    counts = {}
    for ev in events:
        if ev.get("cat") == "kernel" and ev.get("ph") == "X":
            counts[(ev.get("pid"), ev.get("tid"))] = counts.get((ev.get("pid"), ev.get("tid")), 0) + 1
    if not counts:
        return (None, None)
    return max(counts, key=counts.get)


def parse_stage_name(name):
    """Return (phase, ctx_len, gen_len) or None. phase in {'prefill','decode','other'}."""
    m = re.search(r"execute_(?:context|generation)_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)", name)
    if not m:
        return None
    ctx_len, gen_len = int(m.group(2)), int(m.group(4))
    if gen_len > 0:
        phase = "decode"
    elif ctx_len > 0:
        phase = "prefill"
    else:
        phase = "other"
    return (phase, ctx_len, gen_len)


def build_stage_spans(events, dev_tid, dev_pid):
    """Return {name: stage-info}. Each stage name may occur MULTIPLE times
    (vLLM reuses the same stage name across steps of identical shape), so we keep
    per-instance spans. dev_dur_ms = SUM of the per-instance device spans (not the
    union span); stages without a device marker fall back to the CPU-event union."""
    by_name = {}
    for ev in events:
        if ev.get("ph") != "X":
            continue
        name = ev.get("name", "")
        if not STAGE_RE.search(name):
            continue
        info = by_name.setdefault(name, {"phase": None, "dev": [], "cpu": []})
        ts, dur = ev.get("ts", 0), ev.get("dur", 0)
        if (ev.get("pid"), ev.get("tid")) == (dev_pid, dev_tid):
            info["dev"].append((ts, ts + dur))
        else:
            info["cpu"].append((ts, ts + dur))
    spans = {}
    for name, info in by_name.items():
        par = parse_stage_name(name)
        if par is None:
            continue
        dev = sorted(info["dev"])
        cpu = sorted(info["cpu"])
        dev_marked = bool(dev)
        dev_start = min(s for s, _ in (dev or cpu))
        dev_end = max(e for _, e in (dev or cpu))
        # for kernel attribution: per-instance device spans when marked; for
        # unmarked stages (e.g. the CPU-only warming stage) the whole window.
        instances = dev if dev_marked else [(dev_start, dev_end)]
        spans[name] = {
            "phase": par[0],
            "ctx_len": par[1],
            "gen_len": par[2],
            "dev_marked": dev_marked,
            "n_dev_instances": len(dev),
            "n_cpu_instances": len(cpu),
            "dev_instances": instances,
            "dev_start": dev_start,
            "dev_end": dev_end,
            # durations: SUM of per-instance device spans; fallback union for
            # stages with no device marker (e.g. the warming/other stage).
            "dev_dur_ms": (sum(e - s for s, e in dev) / 1000.0) if dev_marked
                          else ((dev_end - dev_start) / 1000.0),
            # cpu_dur_ms = per-instance COMPLETE stage period (the GPU-side
            # annotation duration, ~10ms/stage). The engine thread is blocked
            # waiting on the GPU for the whole stage, so the narrow CPU-side
            # user_annotation (~5.5ms) under-measures CPU occupancy. Unmarked
            # stages fall back to the CPU marker sum.
            "cpu_dur_ms": (sum(e - s for s, e in dev) / 1000.0) if dev_marked
                          else (sum(e - s for s, e in cpu) / 1000.0),
            # CPU-side user_annotation sum kept for reference (what the CPU
            # profiler marker actually measured, excluding the GPU wait).
            "cpu_marker_ms": sum(e - s for s, e in cpu) / 1000.0,
        }
    return spans


def assign_kernel_to_stage(ev, spans):
    ts = ev.get("ts", 0)
    hits = []
    for s, sp in spans.items():
        if not (sp["dev_start"] <= ts <= sp["dev_end"]):
            continue
        # the kernel must fall inside an actual instance, not a gap between
        # repeated instances of the same stage name
        if not any(a <= ts <= b for a, b in sp["dev_instances"]):
            continue
        hits.append((s, sp["dev_end"] - sp["dev_start"]))
    if not hits:
        return None
    # narrowest containing span = the tightest stage
    hits.sort(key=lambda x: x[1])
    return hits[0][0]


def classify(name, rules):
    for cat, pat in rules:
        if re.search(pat, name, re.IGNORECASE):
            return cat
    return "other"


def analyze(events, spans, rules, dev_pid, dev_tid):
    # launch events by correlation id -> cpu launch ts + dur
    launch = {}
    for ev in events:
        if ev.get("cat") == "cuda_runtime" and LAUNCH_NAME_RE.search(ev.get("name", "")):
            corr = (ev.get("args") or {}).get("correlation")
            if corr is not None:
                launch[corr] = ev  # first occurrence wins (CPU-side)
    # kernel buckets: (cat, ts_us, dur_us, launch_dur_us_or_None, name)
    stage_kernels = {s: [] for s in spans}
    stage_kernels["__outside__"] = []
    n_launchless = 0
    for ev in events:
        if ev.get("cat") != "kernel" or ev.get("ph") != "X":
            continue
        name = ev.get("name", "")
        corr = (ev.get("args") or {}).get("correlation")
        ldur = launch[corr].get("dur", 0.0) if corr in launch else None
        if corr not in launch:
            n_launchless += 1
        stage = assign_kernel_to_stage(ev, spans)
        if stage is None:
            stage = "__outside__"
        stage_kernels[stage].append((classify(name, rules), ev.get("ts", 0), ev.get("dur", 0.0), ldur, name))
    return stage_kernels, n_launchless


def agg_rows(stage_kernels, spans, which):
    """Aggregate kernels over all stages whose phase == which."""
    agg = {}  # cat -> [count, total_ms, launch_ms, n_launch, Counter(name)]
    stage_rows = []
    for stage, kws in stage_kernels.items():
        if stage == "__outside__":
            continue
        if spans[stage]["phase"] != which:
            continue
        for cat, ts, dur, ldur, name in kws:
            a = agg.setdefault(cat, [0, 0.0, 0.0, 0, Counter()])
            a[0] += 1
            a[1] += dur / 1000.0
            a[2] += (ldur or 0.0) / 1000.0
            if ldur is not None:
                a[3] += 1
            a[4][short_name(name)] += 1
        stage_rows.append(stage)
    return agg, stage_rows


def fmt_table(agg, title):
    # percentage is relative to the ACTUAL kernel time (sums to ~100%); a
    # stage-duration-based pct was removed because decode stages are mostly
    # GPU bubble and made every category look tiny.
    total = sum(a[1] for a in agg.values())
    rows = []
    for cat, (count, ms, lms, nl, names) in sorted(agg.items(), key=lambda x: -x[1][1]):
        top = ", ".join(f"{n}×{c}" for n, c in names.most_common(4))
        rows.append({
            "category": cat,
            "count": count,
            "n_launch": nl,
            "total_ms": round(ms, 3),
            "avg_ms": round(ms / count, 4) if count else 0.0,
            "pct_of_kernel_sum": round(100 * ms / total, 2) if total else 0.0,
            "cpu_launch_ms": round(lms, 3),
            "top_kernels": top,
        })
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser(description="vLLM V1 trace stage + kernel-category analysis")
    ap.add_argument("trace")
    ap.add_argument("--out", default=None, help="output xlsx path (default: <trace>.stages.xlsx)")
    ap.add_argument("--categories", default=None, help="json list of [name, regex] pairs")
    ap.add_argument("--text", action="store_true", help="print text tables instead of xlsx")
    args = ap.parse_args()

    rules = DEFAULT_CATEGORY_RULES
    if args.categories:
        with open(args.categories) as f:
            rules = [(r[0], r[1]) for r in json.load(f)]

    data = load_trace(args.trace)
    events = data.get("traceEvents", data if isinstance(data, list) else [])
    dev_pid, dev_tid = detect_device_thread(events)
    spans = build_stage_spans(events, dev_tid, dev_pid)
    stage_kernels, n_launchless = analyze(events, spans, rules, dev_pid, dev_tid)

    print(f"trace: {args.trace}")
    print(f"events: {len(events)}  device thread: pid={dev_pid} tid={dev_tid}")
    print(f"stages found: {len(spans)}  kernels not in any stage: {len(stage_kernels['__outside__'])}")
    print(f"kernels without a matching launch event: {n_launchless}")

    prefill = [(s, sp) for s, sp in spans.items() if sp["phase"] == "prefill"]
    decode = [(s, sp) for s, sp in spans.items() if sp["phase"] == "decode"]

    def span_rows(which):
        out = []
        for s, sp in spans.items():
            if sp["phase"] != which:
                continue
            kws = stage_kernels[s]
            ksum = sum(k[2] for k in kws) / 1000.0
            out.append({
                "stage": s,
                "phase": which,
                "dev_marked": sp["dev_marked"],
                "n_dev_instances": sp["n_dev_instances"],
                "n_cpu_instances": sp["n_cpu_instances"],
                "dev_dur_ms": round(sp["dev_dur_ms"], 3),
                "n_kernels": len(kws),
                "kernel_sum_ms": round(ksum, 3),
                "gpu_bubble_ms": round(sp["dev_dur_ms"] - ksum, 3),
                "cpu_dur_ms": round(sp["cpu_dur_ms"], 3),
                "cpu_marker_ms": round(sp["cpu_marker_ms"], 3),
            })
        return out

    # ---- build report dict ----
    report = {
        "meta": {
            "trace": args.trace,
            "n_events": len(events),
            "device": f"pid={dev_pid} tid={dev_tid}",
            "n_stages": len(spans),
            "n_kernels_outside_stage": len(stage_kernels["__outside__"]),
            "n_kernels_no_launch": n_launchless,
            "dev_dur_is_sum_of_instances": "yes (per-instance device spans summed; fallback union when no device marker)",
        },
        "stages": span_rows("prefill") + span_rows("decode") + span_rows("other"),
    }

    for phase, sheet in (("prefill", "prefill_cat"), ("decode", "decode_cat")):
        agg, st = agg_rows(stage_kernels, spans, phase)
        report[sheet] = fmt_table(agg, phase)
        # first stage of the phase as the single representative stage
        if st:
            fs = st[0]
            agg1, _ = agg_rows({fs: stage_kernels[fs]}, spans, phase)
            report[sheet.replace("_cat", "_first")] = fmt_table(agg1, phase)

    report["categories"] = [{"category": c, "regex": r} for c, r in rules]

    if args.text:
        import pprint
        pprint.pprint(report)
        return

    try:
        import pandas as pd
    except ImportError:
        print("pandas not installed; rerun with --text", file=sys.stderr)
        sys.exit(1)

    out = args.out or (args.trace.removesuffix(".gz").removesuffix(".json") + ".xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        for sheet in ("stages", "prefill_cat", "decode_cat", "prefill_first", "decode_first", "categories"):
            rows = report.get(sheet)
            if rows is None:
                continue
            df = pd.DataFrame(rows)
            df.to_excel(xw, sheet_name=sheet[:31], index=False)
        # meta on its own sheet
        pd.DataFrame([{"key": k, "value": v} for k, v in report["meta"].items()]).to_excel(
            xw, sheet_name="meta", index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
