#!/usr/bin/env python3
"""工作流反转的 driver：agent 只写实现，测量/判定/记账由引擎做。

对应 `updates/handoff/HANDOFF-2026-11-28-workflow-inversion-roadmap.md` 的
P3.2（driver）与 P3.3（反馈通道）：

    python3 -I Plugins/ops/engine/iterate.py -C <卡目录> iterate --max-rounds N \
        --agent-cmd "<shell 命令>" [--tag <候选id>] [--dry-run] [--artifacts <根>]

每轮：
  1. 渲染 prompt（第一次是初始 prompt；之后带上一轮 IterationRecord 的**结构化**反馈）；
  2. 把 prompt 从 stdin 喂给 `--agent-cmd`，stdout 只当「改了什么」留痕 ——
     **本文件不从 agent 的输出里解析任何数字**（测量权在引擎）；
  3. 引擎测量：`loop.py run <tag> --bench-std <run根>/results/bench_std.json`；
  4. 引擎审计：`loop.py claim --tag ...`，解析 PASS/FAIL；
  5. 判定 KEEP / REVERT / NEUTRAL：
       - 正确性失败（claim FAIL，且失败项是 accuracy/evidence_kind）→ REVERT
       - claim PASS 且 delta < -1.5% → KEEP
       - 其余 → NEUTRAL（包括「性能回退但结构变了」：不据此 REVERT）
  6. 追加一条 IterationRecord 到 `<run根>/runs/loop/iterate-<ts>.jsonl`。

失败不静默：agent 非零退出 / 超时（默认 1800s）→ 记 REVERT 并继续；
连续 3 次失败（agent、引擎 run、claim 找不到记录都算）→ 停。

`--dry-run` 不调 agent、不跑 run、不写任何文件，只打印将要渲染的 prompt 与前 N 轮计划。

设计约束（刻意不做的事）：
  - 不修改卡片、candidates.yaml、账本；唯一写的是本 driver 自己的 IterationRecord。
  - IterationRecord 刻意**不用**顶层 `tag` / `skv` 字段：`loop.py` 的 `read_records`
    会 glob 同目录的 `*.jsonl`，用 `tag` 会让 `claim --tag <候选id>` 误审到这条
    迭代记录。改用 `candidate` / `variant`，与实验记录的命名空间隔开。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

#: 本 driver 只读引擎模块；不要让 import 顺手改写 `__pycache__`（.pyc 不在交付里）。
sys.dont_write_bytecode = True

ENGINE_DIR = Path(__file__).resolve().parent
LOOP = ENGINE_DIR / "loop.py"

#: KEEP 的门槛（相对 base 的降幅）。噪声底按路线图 P5 取 1.5%。
KEEP_DELTA = -0.015
DEFAULT_TIMEOUT = 1800
MAX_CONSECUTIVE_FAILURES = 3
ACCURACY_MARKERS = ("有精度证据", "evidence_kind", "accuracy")

#: 结构化反馈块里**只放**这些字段（路线图 3.3）。顺序即渲染顺序。
FEEDBACK_FIELDS = (
    "round", "verdict", "cand_ms", "base_ms", "vs_ref", "utilization",
    "gate", "acc", "claim", "refuted_directions",
)


# ------------------------------------------------------------------ 引擎模块加载

def _engine(name):
    """按路径加载 engine/ 下的兄弟模块（`-I` 下脚本目录不在 sys.path）。

    与 loop.py / floor.py 的 `_engine` 同一套键，先挂 sys.modules 再 exec，
    避免模块级缓存分裂成两份。
    """
    key = "_ops_engine_" + name
    if key in sys.modules:
        return sys.modules[key]
    import importlib.util
    spec = importlib.util.spec_from_file_location(key, str(ENGINE_DIR / (name + ".py")))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ 卡片 / 记录

def load_candidates(card):
    """复用 card.load_candidates（含 overlay 合并与归一化）。"""
    return _engine("card").load_candidates(card)


def _candidate_list(cands):
    return [c for c in (cands.get("candidates") or []) if isinstance(c, dict)]


def pick_candidate(cands, tag):
    """按 --tag 选候选；没给就取第一条 pending（再退化到第一条）。"""
    cs = _candidate_list(cands)
    if tag:
        for c in cs:
            if c.get("id") == tag:
                return c
        return None
    for c in cs:
        if str(c.get("status") or "pending") == "pending":
            return c
    return cs[0] if cs else None


def refuted_directions(cands, loop_records):
    """已证伪/已封的方向 + 记录里 falsified=true 的方向，去重保序。"""
    out = []
    for d in (cands.get("directions") or []):
        if not isinstance(d, dict) or not d.get("id"):
            continue
        try:
            n = int(d.get("refutes") or 0)
        except (TypeError, ValueError):
            n = 0
        if str(d.get("status") or "") in ("refuted", "exhausted") or n > 0:
            out.append(str(d["id"]))
    for r in loop_records:
        if r.get("falsified") and r.get("direction"):
            out.append(str(r["direction"]))
    seen, uniq = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def engine_records_dir(card):
    """loop.py 的权威账本目录。

    本仓库的这一版 loop.py 已把运行时数据搬到
    `.artifacts/<arm>/<op>_<run>/runs/loop/`（`loop.artifact_dir`），
    不再是 `<card>/runs/loop`。分辨率依赖 `$OPS_ARM` / `$OPS_RUN_ID` / cwd，
    所以**必须复用引擎自己的函数**，不能在这里复刻一套 —— 复刻就会在
    不同 cwd 下和 run/claim 看到两个账本。
    """
    try:
        return Path(_engine("loop").artifact_dir(Path(card))) / "runs" / "loop"
    except Exception:                                  # noqa: BLE001
        return Path(card) / "runs" / "loop"             # 只装 loop.py 的旧布局兜底


def read_loop_records(card):
    """读引擎账本目录下的 `*.jsonl`（loop.py 的实验记录）。

    返回 [(path, record)]。坏行跳过（与 loop.py::read_records 同口径）。
    """
    d = engine_records_dir(card)
    out = []
    if not d.is_dir():
        return out
    for f in sorted(d.glob("*.jsonl")):
        try:
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                out.append((f, json.loads(line)))
            except json.JSONDecodeError:
                pass
    return out


def read_prev_iteration(record_dir):
    """上一个 IterationRecord：本目录 `iterate-*.jsonl` 的最后一条。

    只认本 driver 写的记录（`record_kind == "iteration"`），避免和实验记录混淆。
    """
    last = None
    d = Path(record_dir)
    if not d.is_dir():
        return None
    for f in sorted(d.glob("iterate-*.jsonl")):
        try:
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("record_kind") == "iteration":
                last = r
    return last


def _append_record(record_dir, filename, rec):
    d = Path(record_dir)
    d.mkdir(parents=True, exist_ok=True)
    f = d / filename
    with f.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return f


# ------------------------------------------------------------------ 反馈 / prompt

def feedback_from(prev):
    """从上一轮 IterationRecord 抽出 3.3 要求的字段。"""
    if not prev:
        return None
    fb = {k: prev.get(k) for k in FEEDBACK_FIELDS}
    fb["acc"] = prev.get("acc") or {}
    fb["refuted_directions"] = prev.get("refuted_directions") or []
    return fb


def _placeholder_feedback(round_no):
    """dry-run 里第 2..N 轮的反馈占位：**明确标成未测量**，不伪造数字。"""
    return {
        "_dry_run_placeholder": True,
        "note": "runtime 时这里会是第 %d 轮的引擎实测记录" % (round_no - 1),
        "round": round_no - 1, "verdict": "PENDING", "cand_ms": None,
        "base_ms": None, "vs_ref": None, "utilization": None, "gate": None,
        "acc": {}, "claim": None, "refuted_directions": [],
    }


def render_prompt(card, cand, run_root, round_no, fb, refuted, what_changed=None):
    """渲染本轮 prompt。fb=None 表示初始 prompt（没有上一条记录）。"""
    tag = (cand or {}).get("id")
    variant = (cand or {}).get("variant")
    direction = (cand or {}).get("direction")
    L = []
    L.append("# 算子实现任务：%s" % Path(card).name)
    L.append("")
    L.append("你只负责**写实现**。测量、判定、记账全部由引擎完成，"
             "你既不需要、也不能在会话里自己测。")
    L.append("")
    L.append("## 硬性约束")
    L.append("")
    L.append("- 不要运行任何 bench（`bench/ab.py`、`time`、`cuda.Event`、任何 `*bench*`）；"
             "引擎会跑并判定。")
    L.append("- 不要修改 `contract.yaml` / `tolerance.json` 里的容差，"
             "不要写 `runs/` 或 `.artifacts/`。")
    L.append("- 每轮只做一个可解释的改动（便于定位是哪一个改动起作用）。")
    L.append("- 标准输出只写「你改了什么」（自然语言）。**不要贴任何耗时/达成率数字** ——"
             "引擎不解析你的输出，只做留痕。")
    L.append("")
    L.append("## 任务")
    L.append("")
    L.append("- 卡目录：`%s`" % Path(card))
    L.append("- 候选：`%s`（变体 `%s`，方向 `%s`）" % (tag, variant, direction))
    L.append("- 工作目录（run 根）：`%s`" % Path(run_root))
    L.append("- 形状档位：见 `%s/shapes.yaml`" % Path(card))
    if (cand or {}).get("means"):
        L.append("- 候选说明：%s" % str(cand["means"]).strip())
    L.append("")
    L.append("## 上一轮（结构化记录）")
    L.append("")
    if fb is None:
        L.append("（无：这是第 1 轮，没有上一轮记录。）")
    else:
        L.append("```json")
        L.append(json.dumps(fb, ensure_ascii=False, indent=2, sort_keys=False))
        L.append("```")
        if what_changed:
            L.append("")
            L.append("上一轮你报告的改动（未解析、仅作参考）：")
            L.append("")
            L.append("> " + "\n> ".join(str(what_changed).splitlines()))
    L.append("")
    L.append("## 已证伪方向（不要再试）")
    L.append("")
    if refuted:
        for x in refuted:
            L.append("- `%s`" % x)
    else:
        L.append("（无）")
    L.append("")
    L.append("## 本轮")
    L.append("")
    L.append("实现候选 `%s` 的一个改动，然后退出。标准输出写「改了什么」。" % tag)
    L.append("")
    return "\n".join(L)


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ 引擎调用

def _tail(text, n=20):
    lines = (text or "").splitlines()
    return "\n".join(lines[-n:])


def run_engine(card, tag, bench_std):
    """`loop.py run <tag> --bench-std <path>`。返回 (rc, stdout+stderr)。"""
    cmd = [sys.executable, "-I", str(LOOP), "-C", str(card), "run", tag,
           "--bench-std", str(bench_std)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
        return p.returncode, (p.stdout or "") + (p.stderr or ""), cmd
    except OSError as e:
        return 127, "%s: %s" % (type(e).__name__, e), cmd


def run_claim(card, tag, variant):
    """`loop.py claim --tag <tag>`；tag 是候选 id，记录里是变体名。

    先按候选 id 问一次（路线图 4 步的字面要求）；若输出了「没有 tag=…」，
    再按变体名重问一次 —— 否则候选 id 与变体名不同的卡永远审计不到。
    返回 (status, failing_checks, no_record, output)。
    """
    attempts = [tag]
    if variant and variant != tag:
        attempts.append(variant)
    out_all = []
    status, fails, no_record = "UNKNOWN", [], False
    for i, t in enumerate(attempts):
        cmd = [sys.executable, "-I", str(LOOP), "-C", str(card), "claim", "--tag", str(t)]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True)
        except OSError as e:
            return "UNKNOWN", [], True, "%s: %s" % (type(e).__name__, e)
        out = (p.stdout or "") + (p.stderr or "")
        out_all.append(out)
        this_no_record = "没有 tag=" in out
        if this_no_record:
            no_record = True
            if i + 1 < len(attempts):
                continue                    # 换变体名再问一次
            status, fails = "FAIL", []
            break
        no_record = False
        fails = [ln.strip()[1:].strip() for ln in out.splitlines()
                 if ln.strip().startswith("✗")]
        if "**PASS**" in out:
            status = "PASS"
        elif "**FAIL**" in out:
            status = "FAIL"
        else:
            status = "PASS" if p.returncode == 0 else "FAIL"
        break
    return status, fails, no_record, "\n".join(out_all)


def floor_metrics(card, cand_ms):
    """达成率 / gate。用 engine/floor.py 的同一套判据，取不到就留 None。"""
    try:
        fl = _engine("floor")
        f = fl.resolve(Path(card))
        t_floor = getattr(f, "t_floor_us", None)
        if not t_floor or not cand_ms or cand_ms <= 0:
            return None, None
        t_act = float(cand_ms) * 1000.0
        remaining = t_act / float(t_floor) - 1.0
        return round(float(t_floor) / t_act, 4), fl.gate_of(remaining)[0]
    except Exception:                                  # noqa: BLE001
        return None, None


def _primary_acc(rec):
    acc = rec.get("acc") or {}
    node = {}
    if isinstance(acc, dict):
        if isinstance(acc.get("1"), dict):
            node = acc["1"]
        else:
            for v in acc.values():
                if isinstance(v, dict):
                    node = v
                    break
    return {
        "max_ulp": node.get("max_ulp"),
        "cosine": node.get("cosine"),
        "evidence_kind": rec.get("evidence_kind"),
    }


def _short_check(s):
    """claim 输出里检查项名与说明之间是大段空格；摘要只留检查项名。"""
    import re
    return re.split(r"\s{2,}", str(s).strip(), maxsplit=1)[0]


def _is_accuracy_failure(fails):
    for f in fails or []:
        low = str(f).lower()
        if any(m.lower() in low for m in ACCURACY_MARKERS):
            return True
    return False


def decide_verdict(delta_frac, claim_status, claim_fails):
    """路线图 3.2 第 4 步。**只按这三条，不加别的规则。**"""
    if claim_status == "FAIL" and _is_accuracy_failure(claim_fails):
        return "REVERT"
    if claim_status == "PASS" and delta_frac is not None and delta_frac < KEEP_DELTA:
        return "KEEP"
    return "NEUTRAL"


# ------------------------------------------------------------------ 一轮

def _record_ref(path, rec):
    return {
        "file": str(path),
        "tag": rec.get("tag"),
        "ts": rec.get("ts"),
        "decision": rec.get("decision"),
        "why": rec.get("why"),
        "measure_mode": rec.get("measure_mode"),
        "impl": rec.get("impl"),
    }


def do_round(a, card, cand, run_root, record_dir, filename, round_no, prev):
    """跑一轮。返回 (rec, ok)，ok=False 计入连续失败。"""
    tag = (cand or {}).get("id")
    variant = (cand or {}).get("variant")
    loop_records = [r for _, r in read_loop_records(card)]
    refuted = refuted_directions(load_candidates(card), loop_records)
    fb = feedback_from(prev)
    prompt = render_prompt(card, cand, run_root, round_no, fb, refuted,
                           what_changed=(prev or {}).get("what_changed"))
    psha = sha256_text(prompt)

    base = {
        "record_kind": "iteration",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "card": Path(card).name,
        "round": round_no,
        "max_rounds": a.max_rounds,
        "candidate": tag,
        "variant": variant,
        "prompt_sha256": psha,
        "refuted_directions": refuted,
    }

    # ---- 1) agent ----
    try:
        ap = subprocess.run(a.agent_cmd, shell=True, input=prompt, text=True,
                            capture_output=True, timeout=a.timeout, cwd=str(run_root))
    except subprocess.TimeoutExpired:
        rec = dict(base, verdict="REVERT", cand_ms=None, base_ms=None,
                   vs_ref=None, utilization=None, gate=None, acc={},
                   claim="SKIPPED", what_changed=None,
                   agent={"cmd": a.agent_cmd, "rc": None, "timed_out": True,
                          "timeout_s": a.timeout, "stderr_tail": ""},
                   record=None, error="agent timeout after %ss" % a.timeout)
        _append_record(record_dir, filename, rec)
        print("[round %d] REVERT  cand=- base=- claim=SKIPPED  (agent 超时 %ss)"
              % (round_no, a.timeout), flush=True)
        return rec, False
    what_changed = ap.stdout or ""
    if ap.returncode != 0:
        rec = dict(base, verdict="REVERT", cand_ms=None, base_ms=None,
                   vs_ref=None, utilization=None, gate=None, acc={},
                   claim="SKIPPED", what_changed=what_changed,
                   agent={"cmd": a.agent_cmd, "rc": ap.returncode, "timed_out": False,
                          "stderr_tail": _tail(ap.stderr)})
        rec["record"] = None
        rec["error"] = "agent exited rc=%d" % ap.returncode
        _append_record(record_dir, filename, rec)
        print("[round %d] REVERT  cand=- base=- claim=SKIPPED  (agent rc=%d)"
              % (round_no, ap.returncode), flush=True)
        if ap.stderr:
            print("  stderr: " + _tail(ap.stderr, 5).replace("\n", "\n          "),
                  flush=True)
        return rec, False

    # ---- 2) 引擎测量 ----
    before = len(read_loop_records(card))
    # 交付物按 opbench 契约放**调用者 cwd**（C 臂里就是 attempt 目录），
    # 不写卡目录 —— P2 要求卡目录不再出现运行时数据。
    bench_std = Path.cwd() / "results" / "bench_std.json"
    rc, run_out, run_cmd = run_engine(card, tag, bench_std)
    after = read_loop_records(card)
    new = after[before:] if len(after) >= before else []
    match_set = {x for x in (tag, variant) if x}
    mine = [rec for _, rec in new if rec.get("tag") in match_set]
    rec_engine = mine[-1] if mine else None

    if rc != 0 or rec_engine is None:
        why = ("引擎 run rc=%d" % rc) if rc != 0 else "引擎 run 没产出本候选的记录"
        rec = dict(base, verdict="REVERT", cand_ms=None, base_ms=None,
                   vs_ref=None, utilization=None, gate=None, acc={},
                   claim="SKIPPED", what_changed=what_changed,
                   agent={"cmd": a.agent_cmd, "rc": ap.returncode, "timed_out": False,
                          "stderr_tail": _tail(ap.stderr)},
                   record=None,
                   command=" ".join(shlex.quote(x) for x in run_cmd),
                   bench_std=str(bench_std),
                   error=why, engine_tail=_tail(run_out))
        _append_record(record_dir, filename, rec)
        print("[round %d] REVERT  cand=- base=- claim=SKIPPED  (%s)" % (round_no, why),
              flush=True)
        return rec, False

    cand_ms = rec_engine.get("cand_ms")
    base_ms = rec_engine.get("base_ms")
    delta_frac = None
    if isinstance(cand_ms, (int, float)) and isinstance(base_ms, (int, float)) and base_ms:
        delta_frac = (cand_ms - base_ms) / base_ms
    vs_ref = (base_ms / cand_ms) if (isinstance(cand_ms, (int, float)) and cand_ms
                                     and isinstance(base_ms, (int, float))) else None
    util, gate = floor_metrics(card, cand_ms)

    # ---- 3) 引擎审计 ----
    claim_status, claim_fails, no_record, claim_out = run_claim(card, tag, variant)
    if no_record:
        rec = dict(base, verdict="REVERT", cand_ms=cand_ms, base_ms=base_ms,
                   vs_ref=round(vs_ref, 4) if vs_ref else None,
                   delta_pct=(round(delta_frac * 100.0, 2) if delta_frac is not None else None),
                   utilization=util, gate=gate, acc=_primary_acc(rec_engine),
                   claim="FAIL", claim_reasons=[], claim_note="claim 找不到记录",
                   what_changed=what_changed,
                   agent={"cmd": a.agent_cmd, "rc": ap.returncode, "timed_out": False,
                          "stderr_tail": ""},
                   record=_record_ref(path_of(new, rec_engine), rec_engine),
                   command=" ".join(shlex.quote(x) for x in run_cmd),
                   bench_std=str(bench_std), impl=rec_engine.get("impl"),
                   error="claim: 没有记录")
        _append_record(record_dir, filename, rec)
        print("[round %d] REVERT  cand=%s base=%s claim=FAIL(no record)"
              % (round_no, _f(cand_ms), _f(base_ms)), flush=True)
        return rec, False

    verdict = decide_verdict(delta_frac, claim_status, claim_fails)
    rec = dict(base, verdict=verdict, cand_ms=cand_ms, base_ms=base_ms,
               vs_ref=round(vs_ref, 4) if vs_ref else None,
               delta_pct=(round(delta_frac * 100.0, 2) if delta_frac is not None else None),
               utilization=util, gate=gate, acc=_primary_acc(rec_engine),
               claim=claim_status, claim_reasons=claim_fails,
               what_changed=what_changed,
               agent={"cmd": a.agent_cmd, "rc": ap.returncode, "timed_out": False,
                      "stderr_tail": _tail(ap.stderr)},
               record=_record_ref(path_of(new, rec_engine), rec_engine),
               command=" ".join(shlex.quote(x) for x in run_cmd),
               bench_std=str(bench_std), impl=rec_engine.get("impl"),
               engine_decision=rec_engine.get("decision"))
    _append_record(record_dir, filename, rec)
    suffix = ""
    if claim_status == "FAIL" and claim_fails:
        suffix = "(%s)" % ",".join(_short_check(f) for f in claim_fails)
    print("[round %d] %-7s cand=%s base=%s claim=%s%s"
          % (round_no, verdict, _f(cand_ms), _f(base_ms), claim_status, suffix),
          flush=True)
    return rec, True


def path_of(new_pairs, rec):
    for p, r in new_pairs:
        if r is rec:
            return p
    return None


def _f(v):
    return "%.4fms" % v if isinstance(v, (int, float)) else "-"


# ------------------------------------------------------------------ dry-run

def dry_run(a, card, cand, run_root, record_dir):
    prev = read_prev_iteration(record_dir)
    loop_records = [r for _, r in read_loop_records(card)]
    refuted = refuted_directions(load_candidates(card), loop_records)
    tag = (cand or {}).get("id")
    variant = (cand or {}).get("variant")
    # 交付物按 opbench 契约放**调用者 cwd**（C 臂里就是 attempt 目录），
    # 不写卡目录 —— P2 要求卡目录不再出现运行时数据。
    bench_std = Path.cwd() / "results" / "bench_std.json"

    print("== iterate 计划（dry-run，不调 agent、不跑 run、不写文件）==")
    print("card      : %s" % Path(card))
    print("candidate : %s (variant=%s, direction=%s)"
          % (tag, variant, (cand or {}).get("direction")))
    print("run root  : %s" % Path(run_root))
    print("records   : %s" % (Path(record_dir) / "iterate-<ts>.jsonl"))
    print("bench_std : %s" % bench_std)
    print("agent cmd : %s   (cwd=%s, stdin=prompt, timeout=%ss)"
          % (a.agent_cmd, Path(run_root), a.timeout))
    print("max rounds: %d" % a.max_rounds)
    print("prev iter : %s" % ("round %s / %s" % (prev.get("round"), prev.get("verdict"))
                              if prev else "none"))
    print("refuted   : %s" % (", ".join(refuted) if refuted else "(none)"))
    print()
    for r in range(1, a.max_rounds + 1):
        fb = feedback_from(prev) if r == 1 else _placeholder_feedback(r)
        prompt = render_prompt(card, cand, run_root, r, fb, refuted,
                               what_changed=(prev or {}).get("what_changed"))
        print("--- round %d plan ---" % r)
        print("1. render prompt (sha256=%s)" % sha256_text(prompt)[:16])
        print("2. pipe prompt -> agent (stdout = what_changed, 不解析数字)")
        print("3. %s -I %s -C %s run %s --bench-std %s"
              % (sys.executable, LOOP, Path(card), tag, bench_std))
        print("4. %s -I %s -C %s claim --tag %s%s"
              % (sys.executable, LOOP, Path(card), tag,
                 (" (fallback --tag %s)" % variant) if variant and variant != tag else ""))
        print("5. verdict: REVERT if claim FAIL(accuracy/evidence_kind); "
              "KEEP if claim PASS and delta < %.1f%%; else NEUTRAL" % (KEEP_DELTA * 100))
        print("6. append IterationRecord -> %s" % (Path(record_dir) /
                                                   "iterate-<ts>.jsonl"))
        print()
        print("--- round %d prompt ---" % r)
        if r > 1:
            print("(dry-run 未执行 round %d，反馈块为占位；runtime 时是真实的"
                  " IterationRecord)" % (r - 1))
        print(prompt)
        print()
    return 0


# ------------------------------------------------------------------ CLI

def cmd_iterate(a):
    card = Path(a.card).resolve()
    if not card.is_dir():
        print("错误：卡目录不存在：%s" % card, file=sys.stderr)
        return 2
    cands = load_candidates(card)
    cand = pick_candidate(cands, a.tag)
    if cand is None:
        print("错误：%s 里没有候选%s"
              % (card / "candidates.yaml", (" %r" % a.tag) if a.tag else ""),
              file=sys.stderr)
        return 2
    if not cand.get("variant"):
        print("错误：候选 %r 没有声明 variant（loop.py run 会拒绝，"
              "先在 candidates.yaml 补 `variant:`）" % cand.get("id"), file=sys.stderr)
        return 2

    run_root = Path(a.artifacts).resolve() if a.artifacts else card
    # P2 之后账本在 .artifacts/<arm>/<op>_<id>/runs/loop/，必须与 run/claim 同源；
    # 直接写 <card>/runs/loop 会让 iterate 的记录和引擎的记录分裂成两处，
    # 而且卡目录会被写脏（P2 的 C2 就是查这个）。
    record_dir = engine_records_dir(card)

    if a.dry_run:
        return dry_run(a, card, cand, run_root, record_dir)

    # 非 dry-run：run 根必须存在（agent 要在里面工作）
    run_root.mkdir(parents=True, exist_ok=True)
    record_dir.mkdir(parents=True, exist_ok=True)
    filename = "iterate-%s.jsonl" % time.strftime("%Y%m%dT%H%M%S")

    print("iterate：card=%s candidate=%s variant=%s rounds=%d"
          % (card.name, cand.get("id"), cand.get("variant"), a.max_rounds), flush=True)
    print("agent=%s cwd=%s timeout=%ss records=%s"
          % (a.agent_cmd, run_root, a.timeout, record_dir / filename), flush=True)

    consecutive_failures = 0
    prev = read_prev_iteration(record_dir)
    for r in range(1, a.max_rounds + 1):
        rec, ok = do_round(a, card, cand, run_root, record_dir, filename, r, prev)
        prev = rec
        if ok:
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print("连续 %d 次失败，停止（fail-closed）。记录：%s"
                      % (consecutive_failures, record_dir / filename), flush=True)
                return 1
    print("完成 %d 轮。记录：%s" % (a.max_rounds, record_dir / filename), flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="iterate.py",
        description="工作流反转 driver：agent 写实现，引擎测量/判定/记账（P3.2/P3.3）。")
    ap.add_argument("-C", "--card", required=True,
                    help="算子卡目录（含 candidates.yaml）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    it = sub.add_parser(
        "iterate", help="多轮：agent 改实现 -> 引擎 run -> claim -> 判定 -> 写记录")
    it.add_argument("--max-rounds", type=int, required=True, metavar="N",
                    help="最多跑 N 轮")
    it.add_argument("--agent-cmd", required=True, metavar="CMD",
                    help="agent 子进程命令（prompt 从 stdin 喂入，stdout 只留痕）")
    it.add_argument("--tag", default=None,
                    help="候选 id（默认取第一条 pending 候选）")
    it.add_argument("--dry-run", action="store_true",
                    help="不调 agent、不跑 run、不写文件，只打印 prompt 与前 N 轮计划")
    it.add_argument("--artifacts", default=None, metavar="ROOT",
                    help="run 根（默认卡目录）；记录写 <ROOT>/runs/loop/，"
                         "bench_std 写 <ROOT>/results/bench_std.json")
    it.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, metavar="S",
                    help="单轮 agent 超时秒数（默认 %d）" % DEFAULT_TIMEOUT)
    a = ap.parse_args(argv)
    if a.cmd == "iterate":
        return cmd_iterate(a)
    ap.error("未知子命令 %r" % a.cmd)
    return 2


if __name__ == "__main__":
    sys.exit(main())
