#!/usr/bin/env python3
"""算子优化循环的驱动器：队列 → 单实验 → 机器判定 → 落盘记账。

这不是一个「自动调优器」。候选假设仍然要人/模型来想；本脚本负责的是
**不让「我觉得快了」蒙混过关**，以及**不重复探索已经证伪的方向**。

    python3 loop.py -C <算子卡目录> status
    python3 loop.py -C <算子卡目录> queue
    python3 loop.py -C <算子卡目录> run maxn32 [--skv 32768 131072]
    python3 loop.py -C <算子卡目录> report
    python3 loop.py -C <算子卡目录> impl-broken <tag> --why "锚点只改到一个 body，算错"
    python3 loop.py -C <算子卡目录> direction-refuted <tag> --why "机制：合并 buffer ⇒ 生命周期重叠 ⇒ 溢出"
    python3 loop.py -C <算子卡目录> direction-refuted --direction fuse-kv-splits --why "结构上不可能"
    python3 loop.py -C <算子卡目录> verify --tag <tag>

完成判定由**外部命令**裁决，不由模型口述：
  候选的 `done_when`（一条 shell 命令）返回 0 ⇒ `verify` 才把 status 改成 adopted；
  非 0 ⇒ 保持原状态，打印输出尾部 20 行并记 `done_when_rc`。
  `adopt` 是 `verify` 的旧名（等价），保留只为兼容脚本——它同样要跑 done_when，
  不存在「不跑命令就 adopted」的路径。

失败分两类，**不要混**（混了就等于白丢一条路）：
  impl-broken        这次实现没写成 / 没跑对（编译不过、锚点没落上、算错）。
                     留在 optimize 修；**不消耗方向预算**；连挂 3 次该实现判死。
  direction-refuted  跑通了，但结论是「这条攻击面不行」，且 why 写清了会连带杀掉哪一族。
                     回 op-constrain 重定约束；同一方向证伪 2 次 ⇒ 封方向，
                     `run` 拒绝再受理该方向的任何候选。

卡片需要提供（其余由本脚本负责）：
  candidates.yaml      方向（directions）+ 候选队列（candidates）+ 已判死清单（refuted）
                       （见 op-optimize/SKILL.md §三 的 schema）
  bench/ab.py          卡**自带**的 A/B 适配器（可选）：同进程交替 A/B，支持 --json，输出见
                       ADAPTER_AB。卡片没提供时 `run` 明确拒绝（fail-closed），`status`/`queue` 照常
  bench/acc_probe.py   卡自带的精度适配器（可选；没有就跳过精度门）
  contract.yaml        精度容差（可选；没有就用 --tol 默认值）

状态全部落在 <卡片>/runs/loop/（已被仓库 .gitignore 的 `**/runs/` 覆盖），
所以上下文丢了不会丢进度：`status` 一跑就能接着干。
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ADAPTER_AB = """bench/ab.py --json 必须输出：
{"shape": {...},
 "runs": [{"skv": 32768, "base_ms": 0.78, "base_mad": 0.004, "fm_ms": 0.40,
           "variants": [{"tag": "maxn32", "ms": 0.69, "mad": 0.003,
                         "delta_pct": -11.7, "maxdiff": 1.9e-3, "rel": 4e-5}]}]}
（base 必须在每轮被重测；只报 base 一次的实现会把漂移算成收益）"""

ADAPTER_ACC = """bench/acc_probe.py --json 必须输出：
{"qscale": {"1": {"ref_absmax": 3.1, "rows": [{"tag": "base", "max_abs": 1.0e-3, "nonfinite": 0},
                                             {"tag": "maxn32", "max_abs": 1.1e-3, "nonfinite": 0}]}}}
"""

OK, WARN = "\033[32m", "\033[33m"

# 两个上限是**常量不是开关**：能拿参数绕过去的规则等于没规则。
MAX_IMPL_FAILS = 3          # 同一实现连挂 N 次 ⇒ 这个实现判死（方向预算不动）
MAX_DIRECTION_REFUTES = 2   # 同一方向证伪 N 次 ⇒ 封方向（拒绝再受理同类候选）

# verify 的口径（同为常量，不给开关去绕）：
VERIFY_TIMEOUT_S = 3600     # done_when 是外部命令，最长给 1 小时（跑真 bench 也够）
VERIFY_TAIL_LINES = 20      # 失败时只回显输出尾部 20 行，别把整份日志灌进上下文

REPO_ROOT = Path(__file__).resolve().parents[4]


def _c(s, color):
    return s if not sys.stdout.isatty() else color + s + "\033[0m"


# ---------------------------------------------------------------- 卡片读取

def load_candidates(card):
    p = card / "candidates.yaml"
    if not p.exists():
        return {"defaults": {}, "directions": [], "candidates": [], "refuted": []}
    import yaml
    d = yaml.safe_load(p.read_text()) or {}
    d.setdefault("defaults", {})
    d.setdefault("directions", [])
    d.setdefault("candidates", [])
    d.setdefault("refuted", [])
    return d


def direction_table(cands):
    """方向 → {refutes, status, exhausted}。计数只有这一个权威点（本段 yaml）。"""
    out = {}
    for d in cands.get("directions") or []:
        if not isinstance(d, dict) or not d.get("id"):
            continue
        refutes = int(d.get("refutes") or 0)
        status = str(d.get("status") or "open")
        out[str(d["id"])] = {
            "id": str(d["id"]),
            "statement": str(d.get("statement") or ""),
            "refutes": refutes,
            "status": status,
            "exhausted": status == "exhausted" or refutes >= MAX_DIRECTION_REFUTES,
        }
    return out


def budget_problems(cands, tags=None):
    """方向预算的前置条件。返回问题清单，空 = 预算可计、没封。

    三条其实是同一件事：**方向预算对这个候选不可计，或它属于已封的方向**
    ⇒ 2 次上限保护不到它 ⇒ 拒绝跑。宁可拦住，也不许「规则对某个候选不生效」。
    """
    table = direction_table(cands)
    probs = []
    for c in cands.get("candidates") or []:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("id") or "?")
        if tags is not None and cid not in tags:
            continue
        if str(c.get("status") or "pending") != "pending":
            continue
        if len(c.get("fails") or []) >= MAX_IMPL_FAILS:
            probs.append("候选 %s 的实现已连挂 %d 次（判死线），它该在 refuted 里"
                         % (cid, MAX_IMPL_FAILS))
            continue
        d = c.get("direction")
        if not d:
            probs.append("候选 %s 没声明 direction：方向预算计不了（%d 次上限对它是空转）"
                         % (cid, MAX_DIRECTION_REFUTES))
            continue
        info = table.get(str(d))
        if info is None:
            probs.append("候选 %s 的 direction=%s 不在 directions 段里（拼错？预算同样计不了）"
                         % (cid, d))
        elif info["exhausted"]:
            probs.append("候选 %s 属于方向 %s，该方向已证伪 %d 次（%s）⇒ 回 op-constrain "
                         "重定约束，别再在这个方向上生成候选"
                         % (cid, d, info["refutes"], info["status"]))
    return probs


def done_when_problems(cands, tags=None):
    """`status: adopted` 的候选必须带 `done_when`（一条 shell 命令）。

    理由：adopted 是「做完了」的断言，而**模型说做完不算数**——它必须能被一个
    单独的 evaluator 跑出来（`loop.py verify --tag <id>`）。没有命令的 adopted
    等于没有参照物的自我批准，所以 `run` 拒绝受理它。

    只看 adopted；pending 候选还没有完成判定，不要求（`verify` 会另拦）。
    """
    probs = []
    for c in cands.get("candidates") or []:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("id") or "?")
        if tags is not None and cid not in tags:
            continue
        if str(c.get("status") or "pending") != "adopted":
            continue
        if not str(c.get("done_when") or "").strip():
            probs.append("候选 %s 已是 adopted 但没写 done_when：完成判定没有外部裁决点"
                         "（模型说完成不算数）⇒ 补一条命令，再用 verify 重新裁决" % cid)
    return probs


def load_contract_tol(card, default=2e-3):
    p = card / "contract.yaml"
    if not p.exists():
        return default
    try:
        import yaml
        c = yaml.safe_load(p.read_text()) or {}
    except Exception:
        return default
    for path in (("tolerance", "vs_fp32_ref", "max_abs"), ("tolerance", "max_abs")):
        node = c
        for k in path:
            node = node.get(k) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, (int, float)):
            return float(node)
    return default


def db_dir(card):
    d = card / "runs" / "loop"
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_records(card):
    out = []
    for f in sorted(db_dir(card).glob("*.jsonl")):
        for line in f.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def append_record(card, rec):
    f = db_dir(card) / (time.strftime("%Y%m%d") + ".jsonl")
    with f.open("a") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- 判定

def decide(base, cand, acc_base, acc_cand, tol, primary, stress_mult):
    """返回 (decision, why)。

    decision ∈ ACCEPTED_* / REJECTED_* / INCONCLUSIVE。

    精度门分两层，**不能混**（这是本项目踩出来的坑）：
      - 主口径（primary qscale = 生产 logit 尺度）：硬门，max_abs 必须 ≤ contract 容差。
      - 压力档（更高 qscale，故意把 softmax 拉尖）：只判**相对退化的倍数**
        `cand_rel / base_rel`。拿绝对容差去判压力档是错的 —— base 自己在 qscale=10
        就有 1.55e-2，会把正确改动一起否掉。压力档只报警，报警就必须补端到端。
    """
    if "error" in cand:
        return "ERROR", cand["error"]

    warns = []
    if acc_cand:
        nf = sum(r.get("nonfinite", 0) for r in acc_cand.values())
        if nf:
            return "REJECTED_ACCURACY", "出现 nonfinite（%s）" % ", ".join(
                "qscale=%s:%d" % (q, r["nonfinite"]) for q, r in acc_cand.items() if r["nonfinite"])
        p = acc_cand.get(primary)
        if p and p.get("max_abs", 0.0) > tol:
            return "REJECTED_ACCURACY", "主口径 qscale=%s 对 fp32 参考 max_abs=%.2e > 容差 %.1e" % (
                primary, p["max_abs"], tol)
        for q, r in acc_cand.items():
            if q == primary:
                continue
            b = (acc_base or {}).get(q)
            if b and b.get("rel"):
                k = r.get("rel", 0.0) / b["rel"]
                if k > stress_mult:
                    warns.append("qscale=%s 相对退化 %.1f×（%.2e vs base %.2e）"
                                 % (q, k, r.get("rel", 0.0), b["rel"]))

    b = {"median_ms": base["median_ms"], "mad_ms": base.get("mad_ms", 0.0)}
    c = {"median_ms": cand["ms"], "mad_ms": cand.get("mad", 0.0)}
    sig, ratio, why = _significant(b, c)
    tail = ("；⚠ " + "；".join(warns) + " ⇒ 压力档有退化，端到端必须验") if warns else ""
    if not sig:
        if ratio is not None and ratio < 1.0:
            return "INCONCLUSIVE", why + "（方向对但没过噪声门，别当收益写）"
        return "REJECTED_NO_GAIN", why

    if cand.get("maxdiff", 0.0) == 0.0:
        return "ACCEPTED_BITEXACT", "显著加速 %.1f%%，且输出逐位一致%s" % (-cand.get("delta_pct", 0.0), tail)
    return "ACCEPTED_NUMERICS", ("显著加速 %.1f%%，但**非逐位一致**（maxdiff=%.2e）⇒ 必须过端到端质量门%s"
                                 % (-cand.get("delta_pct", 0.0), cand.get("maxdiff", 0.0), tail))


def _significant(base, cand, k=3.0):
    """与 scripts/opbench.py:significant 同义（这里内联，避免卡片路径依赖）。"""
    b, bmad = base["median_ms"], base.get("mad_ms", 0.0) or 0.0
    c, cmad = cand["median_ms"], cand.get("mad_ms", 0.0) or 0.0
    if not b or not c:
        return False, None, "缺少数值"
    ratio = c / b
    noise = k * max(bmad, cmad)
    delta = abs(c - b)
    if delta <= noise:
        return False, ratio, "在噪声内（|Δ|=%.4g <= %.0f*MAD=%.4g）" % (delta, k, noise)
    return ratio < 1.0, ratio, "显著：ratio=%.4f，|Δ|=%.4g > %.0f*MAD=%.4g" % (ratio, delta, k, noise)


# ---------------------------------------------------------------- 子命令

SURFACES_PATH = Path(__file__).resolve().parent / "references" / "surfaces.yaml"


def load_surfaces():
    """读攻击面表。缺文件返回空表（不报错——老卡片环境可能没有）。"""
    if not SURFACES_PATH.exists():
        return []
    import yaml
    d = yaml.safe_load(SURFACES_PATH.read_text(encoding="utf-8")) or []
    return [s for s in d if isinstance(s, dict) and s.get("id")]


def cmd_enumerate(a):
    """列出可用攻击面，并标出「已被方向预算封掉」的那些。

    这是给人和 agent 看的候选菜单：direction 只能从这张表里挑，
    不再允许凭空发明。已封的方向直接标出来，避免白跑。

    --new 只打印可以填进 candidates.yaml 的 YAML 片段。
    """
    surfaces = load_surfaces()
    if not surfaces:
        print(_c("找不到 %s" % SURFACES_PATH, WARN), file=sys.stderr)
        return 2

    cands = load_candidates(a.card)
    table = direction_table(cands)
    pending_dirs = {str(c.get("direction")) for c in (cands.get("candidates") or [])
                    if isinstance(c, dict) and str(c.get("status") or "pending") == "pending"}

    rows = []
    for s in surfaces:
        sid = str(s["id"])
        info = table.get(sid)
        if info and info["exhausted"]:
            state, note = "SEALED", "已证伪 %d 次，回 L1 重定约束" % info["refutes"]
        elif sid in pending_dirs:
            state, note = "QUEUED", "已有 pending 候选挂在这个方向"
        elif info and info["refutes"]:
            state, note = "OPEN*", "已证伪 %d 次（再 1 次封方向）" % info["refutes"]
        else:
            state, note = "OPEN", ""
        rows.append((state, sid, s.get("class", ""), s.get("cost", ""),
                     s.get("law", ""), s.get("statement", ""), note))

    if a.new:
        for state, sid, cls, cost, law, stmt, _ in rows:
            if state.startswith("OPEN"):
                print("- id: %s" % sid)
                print("  direction: %s" % sid)
                print("  status: pending")
                print("  # %s（%s/%s  %s）" % (stmt, cls, cost, law))
        return 0

    print("%-7s %-18s %-12s %-4s %-4s %s" %
          ("状态", "surface", "class", "cost", "law", "说明"))
    for state, sid, cls, cost, law, stmt, note in rows:
        color = {"OPEN": "", "OPEN*": WARN, "QUEUED": "", "SEALED": WARN}.get(state, "")
        line = "%-7s %-18s %-12s %-4s %-4s %s" % (state, sid, cls, cost, law, stmt)
        if note:
            line += "  ← " + note
        print(_c(line, color) if color else line)

    open_n = sum(1 for r in rows if r[0].startswith("OPEN"))
    print("\n可挑 %d 条 / 共 %d 条；已封 %d 条。"
          % (open_n, len(rows), sum(1 for r in rows if r[0] == "SEALED")))
    print("下一步：把选中的 id 填进 candidates.yaml 的 direction（可加 --new 直接出片段）")
    return 0


def cmd_status(a):
    card, cands, recs = a.card, load_candidates(a.card), read_records(a.card)
    pend = [c for c in cands["candidates"]
            if isinstance(c, dict) and c.get("status", "pending") == "pending"]
    print("卡片      %s" % card)
    print("候选      待做 %d / 已判死 %d / 共记录 %d 条实验"
          % (len(pend), len(cands["refuted"]), len(recs)))
    table = direction_table(cands)
    if table:
        print("方向      " + " · ".join(
            "%s %d/%d%s" % (k, v["refutes"], MAX_DIRECTION_REFUTES,
                            "（已封）" if v["exhausted"] else "")
            for k, v in sorted(table.items())))
    for prob in budget_problems(cands):
        print(_c("⚠ " + prob, WARN))
    if recs:
        last = recs[-1]
        print("最近一次  %s %s → %s" % (last.get("ts"), last.get("tag"), last.get("decision")))
    best = {}
    for r in recs:
        if r.get("decision", "").startswith("ACCEPTED") and r.get("decision") != "ACCEPTED_NUMERICS_PENDING_E2E":
            key = r["skv"]
            if key not in best or r["cand_ms"] < best[key]["cand_ms"]:
                best[key] = r
    for skv in sorted(best):
        r = best[skv]
        print("最强      SKV=%-7d %-14s %.4f（base %.4f，%+.1f%%）"
              % (skv, r["tag"], r["cand_ms"], r["base_ms"], r["delta_pct"]))
    runnable = [c for c in pend if not budget_problems(cands, tags={c.get("id")})]
    if runnable:
        print("\n下一步：")
        for c in sorted(runnable, key=lambda x: x.get("priority", 99))[:3]:
            print("  [%s] %s — %s" % (c.get("priority", "-"), c.get("id"), c.get("hypothesis", "")))
    elif pend:
        print("\n下一步：没有可跑的候选 —— 待做的 %d 条都被方向预算拦住了（见上）。" % len(pend))


def cmd_queue(a):
    d = load_candidates(a.card)
    done = {r["tag"] for r in read_records(a.card)}
    table = direction_table(d)
    if table:
        print("# 方向预算（证伪 %d 次封方向 ⇒ 该方向的候选一律不再受理）" % MAX_DIRECTION_REFUTES)
        for k, v in sorted(table.items()):
            print("%-26s %d/%d  %-9s %s" % (k, v["refutes"], MAX_DIRECTION_REFUTES,
                                           v["status"], v["statement"]))
        print()
    print("# 待做（按 priority）")
    for c in sorted([c for c in d["candidates"]
                     if isinstance(c, dict) and c.get("status", "pending") == "pending"],
                    key=lambda x: x.get("priority", 99)):
        mark = "已跑过" if c["id"] in done else ""
        print("%-3s %-22s [%s] %s" % (c.get("priority", "-"), c["id"],
                                      c.get("direction", "无方向"), c.get("hypothesis", "")))
        for k in ("expect", "why", "evidence"):
            if c.get(k):
                print("      %-8s %s" % (k, c[k]))
        fails = c.get("fails") or []
        if fails:
            print("      %-8s %d/%d 次实现失败；最近：%s"
                  % ("fails", len(fails), MAX_IMPL_FAILS, fails[-1]))
        if mark:
            print("      %-8s %s" % ("note", mark))
    if d["refuted"]:
        print("\n# 已判死（别重开；机制比结论重要）")
        for r in d["refuted"]:
            print("%-22s %-10s %s" % (r["id"], r.get("kind", "impl"), r.get("why", "")))
    dw = done_when_problems(d)
    if dw:
        print()
        for prob in dw:
            print(_c("⚠ " + prob, WARN))
    probs = budget_problems(d)
    if probs:
        print()
        for prob in probs:
            print(_c("⚠ " + prob, WARN))
        return 2
    return 0


def cmd_run(a):
    card = a.card
    cands = load_candidates(card)
    refuted = {r["id"]: r for r in cands["refuted"]}
    declared = {c.get("id"): c for c in cands["candidates"] if isinstance(c, dict)}
    forced = set()
    for tag in a.tags:
        if tag in refuted:
            kind = refuted[tag].get("kind", "impl")
            label = {"direction": "方向已证伪", "impl": "实现已判死"}.get(kind, "已判死")
            if not a.force:
                print(_c("拒绝重跑 %s：%s —— %s" % (tag, label, refuted[tag].get("why", "")), WARN))
                print("（确实有理由就加 --force，并在记录里写清楚为什么这次不一样）")
                return 2
            # --force 是**显式否决**：这个 tag 已不在队列里，挂不上方向，就别拿方向预算
            # 去拦它；记录里 forced=true 留痕，事后能查出是谁把死马又跑了一遍
            forced.add(tag)
        elif tag not in declared:
            print(_c("拒绝跑 %s：不在 candidates.yaml 的队列里。先把它写进队列"
                     "（带 direction + expect），再跑 —— 没登记的实验记账无从谈起。" % tag, WARN))
            return 2
    # 已 adopted 但没写 done_when ⇒ 它的「完成」没有任何外部裁决点，不许再跑
    for prob in done_when_problems(cands, tags=set(a.tags)):
        print(_c("拒绝跑：" + prob, WARN))
        return 2
    # 方向预算不可计 / 已封 ⇒ 这个 tag 的「2 次封方向」约束保护不到它，fail-closed
    for prob in budget_problems(cands, tags=set(a.tags) - forced):
        print(_c("拒绝跑：" + prob, WARN))
        return 2
    if forced:
        print(_c("⚠ --force 重跑已判死项：%s（这次不一样的理由必须写进记录）"
                 % ", ".join(sorted(forced)), WARN))

    d = cands["defaults"]
    bench = card / d.get("bench", "bench/ab.py")
    if not bench.exists():
        print(_c("拒绝跑：本卡没有 A/B 适配器 %s（卡片需自带 `bench/ab.py`，契约见本文件 ADAPTER_AB）。"
                 "改用卡片自己的 bench/opcheck 命令跑，再用 `impl-broken` / `direction-refuted` 记账。"
                 % bench, WARN))
        return 2
    skv = a.skv or d.get("skv") or [32768]
    ab_json = db_dir(card) / "_ab.json"
    cmd = [sys.executable, str(bench), *a.tags,
           "--json", str(ab_json), "--rounds", str(a.rounds),
           "--s", str(a.s or d.get("s", 128)),
           "--topk", str(a.topk or d.get("topk", 2048)),
           "--nvalid", str(a.nvalid or d.get("nvalid", 2000)),
           "--heads", str(a.heads or d.get("heads", 64)),
           "--skv", *[str(x) for x in skv]]
    print("$ " + " ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    if rc != 0 or not ab_json.exists():
        print(_c("A/B 失败（rc=%d），没有记录" % rc, WARN))
        return 1
    ab = json.loads(ab_json.read_text())

    acc = {}
    if not a.no_acc and (card / d.get("acc", "bench/acc_probe.py")).exists():
        acc_json = db_dir(card) / "_acc.json"
        cmd = [sys.executable, str(card / d.get("acc", "bench/acc_probe.py")), *a.tags,
               "--json", str(acc_json),
               "--qscale", *[str(x) for x in (a.qscale or d.get("acc_qscale") or [1, 10])],
               "--skv", str(skv[0])]
        print("$ " + " ".join(cmd), flush=True)
        if subprocess.call(cmd) == 0 and acc_json.exists():
            raw = json.loads(acc_json.read_text())
            for qs, node in (raw.get("qscale") or {}).items():
                for row in node.get("rows", []):
                    acc.setdefault(row["tag"], {})[qs] = row

    tol = a.tol or load_contract_tol(card)
    primary = "%g" % float(a.primary_qscale or d.get("acc_primary_qscale") or 1.0)
    stress_mult = float(a.stress_mult or d.get("acc_stress_mult") or 1.5)
    worst = "ACCEPTED_BITEXACT"
    order = ["ERROR", "REJECTED_ACCURACY", "REJECTED_NO_GAIN", "INCONCLUSIVE",
             "ACCEPTED_NUMERICS_PENDING_E2E", "ACCEPTED_BITEXACT"]

    print("\n%-14s %-9s %-8s %s" % ("tag", "skv", "Δ%", "判定"))
    for run in ab["runs"]:
        for v in run["variants"]:
            decision, why = decide(
                {"median_ms": run["base_ms"], "mad_ms": run.get("base_mad", 0.0)},
                v, acc.get("base"), acc.get(v["tag"]), tol, primary, stress_mult)
            if decision == "ACCEPTED_NUMERICS":
                decision = "ACCEPTED_NUMERICS_PENDING_E2E"
            if order.index(decision) < order.index(worst):
                worst = decision
            print("%-14s %-9d %+8.1f %s  %s" % (v["tag"], run["skv"], v.get("delta_pct", 0.0),
                                                decision, why))
            append_record(card, {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "card": card.name,
                "tag": v["tag"], "skv": run["skv"],
                "direction": (declared.get(v["tag"]) or {}).get("direction"),
                "shape": ab.get("shape", {}),
                "base_ms": run["base_ms"], "base_mad": run.get("base_mad", 0.0),
                "fm_ms": run.get("fm_ms"), "fm_ratio": run.get("fm_ratio"),
                "cand_ms": v.get("ms"), "mad": v.get("mad"),
                "delta_pct": v.get("delta_pct"), "ratio_vs_base": v.get("ratio_vs_base"),
                "maxdiff": v.get("maxdiff"), "rel": v.get("rel"),
                "acc": acc.get(v["tag"], {}), "tolerance": tol,
                "decision": decision, "why": why,
                "forced": bool(a.force and v["tag"] in refuted),
                # 完成判定的裁决码：run 不跑 done_when（那是 verify 的事），记 None 让
                # 流水里字段恒在，消费方不必按记录类型分叉。
                "done_when_rc": None,
            })
    print("\n合计判定：%s" % worst)
    if worst.startswith("ACCEPTED_NUMERICS"):
        print("下一步：这个改动不是逐位一致的，必须补端到端质量门（GPQA/HumanEval 一档）才能进生产。")
    elif worst == "ACCEPTED_BITEXACT":
        print("下一步：先给候选补 done_when，再 loop.py verify --tag <tag>（rc=0 才算完成），"
              "然后把它落进生产文件并复测。")
    return 0


def cmd_report(a):
    # 只渲染性能实验记录：verify 记录没有 skv/cand_ms，混进来会把交付表打崩
    recs = [r for r in read_records(a.card) if "skv" in r and "cand_ms" in r]
    if not recs:
        print("还没有记录（先跑 loop.py run <tag>）")
        return 1
    best = {}
    for r in recs:
        key = (r["skv"], r["tag"])
        if key not in best or r["cand_ms"] < best[key]["cand_ms"]:
            best[key] = r
    print("| SKV | tag | base | cand | Δ | vs fm | maxdiff | 判定 |")
    print("|---|---|---|---|---|---|---|---|")
    for (skv, tag), r in sorted(best.items()):
        print("| %d | %s | %.4f | %.4f | %+.1f%% | %s | %.2e | %s |" % (
            skv, tag, r["base_ms"], r["cand_ms"], r["delta_pct"],
            "%.2fx" % r["fm_ratio"] if r.get("fm_ratio") else "-",
            r.get("maxdiff") or 0.0, r["decision"]))
    return 0


def _yaml_scalar(s):
    return json.dumps(str(s), ensure_ascii=False)


def _section_span(text, key):
    """顶格段 `key:` 的 [start, end)。段尾 = 下一个顶格非空行。"""
    m = re.search(r"(?m)^%s:" % re.escape(key), text)
    if m is None:
        return None
    nxt = re.search(r"(?m)^\S", text[m.end():])
    return m.start(), (m.end() + nxt.start()) if nxt else len(text)


def _item_block(text, section, item_id):
    """段内 `  - id: <item_id>` 那一块的绝对 [start, end)。

    边界：下一个 `  - id: ` 或任何顶格非空行。**只在指定段里找**，
    避免 directions / candidates / refuted 三处同名 id 互相误伤。
    """
    sp = _section_span(text, section)
    if sp is None:
        return None
    seg = text[sp[0]:sp[1]]
    m = re.search(r"(?m)^  - id: %s[ \t]*$" % re.escape(item_id), seg)
    if m is None:
        return None
    rest = seg[m.end():]
    nxt = re.search(r"(?m)^(?:  - id: |\S)", rest)
    end = m.end() + (nxt.start() if nxt else len(rest))
    return sp[0] + m.start(), sp[0] + end


def _cand_block(text, tag):
    return _item_block(text, "candidates", tag)


def _write_candidates(p, text):
    """先校验再落盘：写坏了立刻炸，且不覆盖原文件。"""
    import yaml
    try:
        yaml.safe_load(text)
    except Exception as e:
        raise SystemExit("拒绝写入：生成的 YAML 无法解析（%s）。原文件未改动。" % e)
    p.write_text(text)


def _drop_refuted(text, tag):
    """删掉 refuted 段里同 id 的旧条目（保证判死是幂等的）。"""
    blk = _item_block(text, "refuted", tag)
    if blk is None:
        return text
    return text[:blk[0]] + text[blk[1]:]


def _scalar(v):
    """简单标识符不加引号（和手写风格一致），其余走 JSON 字符串。"""
    if isinstance(v, bool) or isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str) and re.match(r"^[A-Za-z_][\w.-]*$", v):
        return v
    return _yaml_scalar(v)


def _render_field(indent, key, value):
    pad = " " * indent
    if isinstance(value, (list, tuple)):
        out = "%s%s:\n" % (pad, key)
        for v in value:
            out += "%s  - %s\n" % (pad, _scalar(v))
        return out
    return "%s%s: %s\n" % (pad, key, _scalar(value))


def _field_span(seg, indent, key):
    """seg 内 `    key:` 这一段的 [start, end)；多行列表值（更深缩进的 `- ...`）一并算进来。"""
    lines = seg.splitlines(keepends=True)
    off = 0
    for i, ln in enumerate(lines):
        if re.match(r"^%s%s:" % (" " * indent, re.escape(key)), ln):
            j = i + 1
            while j < len(lines) and re.match(r"^ {%d,}\S" % (indent + 2), lines[j]):
                j += 1
            return off, off + sum(len(x) for x in lines[i:j])
        off += len(ln)
    return None


def _set_field(text, block, indent, key, value):
    """在 block 指向的条目里设置字段：有就整段替换（含多行值），没有就追加到最后。"""
    if block is None:
        return text
    s, e = block
    seg = text[s:e]
    if not seg.endswith("\n"):
        seg += "\n"
    rendered = _render_field(indent, key, value)
    span = _field_span(seg, indent, key)
    if span:
        seg = seg[:span[0]] + rendered + seg[span[1]:]
    else:
        seg = seg.rstrip("\n") + "\n" + rendered
    return text[:s] + seg + text[e:]


def _to_refuted(text, tag, why, kind, direction=None):
    """把一个条目从 candidates/directions 挪进 refuted —— **已判死只有这一个落点**。"""
    blk = _cand_block(text, tag)
    if blk:
        text = text[:blk[0]] + text[blk[1]:]
    text = _drop_refuted(text, tag)
    body = "  - id: %s\n    kind: %s\n" % (_scalar(tag), _scalar(kind))
    if direction:
        body += "    direction: %s\n" % _scalar(direction)
    body += "    why: %s\n    on: %s\n" % (_yaml_scalar(why), time.strftime("%Y-%m-%d"))
    m = re.search(r"(?m)^refuted:[ \t]*$\n?", text)
    if not m:
        text = text.rstrip("\n") + "\nrefuted:\n" + body
    else:
        text = text[:m.end()] + body + text[m.end():]
    return text.replace("refuted: []\n", "refuted:\n")


def _add_direction(text, dir_id):
    """direction-refuted --direction X 遇到未声明的方向：建条目（证伪事件本身就是事实）。"""
    entry = ("  - id: %s\n    statement: ''\n    status: open\n    refutes: 0\n"
             % _scalar(dir_id))
    sp = _section_span(text, "directions")
    if sp is not None:
        m = re.search(r"(?m)^directions:[ \t]*$", text)
        return text[:m.end() + 1] + entry + text[m.end() + 1:]
    anchor = _section_span(text, "candidates") or _section_span(text, "refuted")
    if anchor is None:
        return text.rstrip("\n") + "\ndirections:\n" + entry
    return text[:anchor[0]] + "directions:\n" + entry + "\n" + text[anchor[0]:]


def cmd_impl_broken(a):
    """实现级失败：留在 optimize 修，**不消耗方向预算**；连挂 N 次该实现判死。"""
    card = a.card
    p = card / "candidates.yaml"
    cands = load_candidates(card)
    cand = next((c for c in cands["candidates"]
                 if isinstance(c, dict) and c.get("id") == a.tag), None)
    if cand is None:
        print(_c("拒绝：%s 不在 candidates 段里（先把它写进队列，再谈失败次数）" % a.tag, WARN))
        return 2
    fails = list(cand.get("fails") or [])
    fails.append("%s %s" % (time.strftime("%Y-%m-%d"), a.why))
    n = len(fails)
    direction = cand.get("direction")
    text = p.read_text()
    text = _set_field(text, _cand_block(text, a.tag), 4, "fails", fails)
    dead = n >= MAX_IMPL_FAILS
    if dead:
        text = _to_refuted(text, a.tag, "连挂 %d 次：%s" % (n, "；".join(fails)), "impl", direction)
    _write_candidates(p, text)
    append_record(card, {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "card": card.name, "tag": a.tag,
        "direction": direction, "decision": "IMPL_BROKEN", "n": n,
        "exhausted": dead, "why": a.why,
    })
    if dead:
        print("%s 连挂 %d 次 ⇒ 这个实现判死（已进 refuted，kind=impl）。" % (a.tag, n))
        if direction:
            print("方向 %s 的预算**没有**被消耗 —— 换个实现重来，别换方向。" % direction)
    else:
        print("%s 记一次实现失败（%d/%d）：修实现；方向预算没动。"
              % (a.tag, n, MAX_IMPL_FAILS))
        if n + 1 >= MAX_IMPL_FAILS:
            print(_c("下一次再挂就判死这个实现。", WARN))
    return 0


def cmd_direction_refuted(a):
    """方向级失败：回 op-constrain 重定约束；证伪 N 次封方向，拒绝再受理同类候选。"""
    card = a.card
    p = card / "candidates.yaml"
    cands = load_candidates(card)
    cand = None
    if a.tag:
        cand = next((c for c in cands["candidates"]
                     if isinstance(c, dict) and c.get("id") == a.tag), None)
    dir_id = a.direction or (cand or {}).get("direction")
    if not dir_id:
        if a.tag and cand is None:
            print(_c("拒绝：%s 不在 candidates 段里，且没给 --direction" % a.tag, WARN))
        else:
            print(_c("拒绝：这个候选没声明 direction，方向预算无从计。"
                     "补 direction: 或直接给 --direction <id>", WARN))
        return 2

    table = direction_table(cands)
    text = p.read_text()
    info = table.get(str(dir_id))
    if info is None:
        text = _add_direction(text, dir_id)
        refutes, status = 0, "open"
        print("directions 段里没有 %s，已建条目（statement 待补）" % dir_id)
    else:
        refutes, status = info["refutes"], info["status"]
    refutes += 1
    if refutes >= MAX_DIRECTION_REFUTES:
        status = "exhausted"
    text = _set_field(text, _item_block(text, "directions", str(dir_id)), 4, "refutes", refutes)
    text = _set_field(text, _item_block(text, "directions", str(dir_id)), 4, "status", status)
    if cand is not None:
        text = _to_refuted(text, a.tag, a.why, "direction", str(dir_id))
    _write_candidates(p, text)
    append_record(card, {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "card": card.name, "tag": a.tag,
        "direction": str(dir_id), "decision": "DIRECTION_REFUTED",
        "refutes": refutes, "exhausted": refutes >= MAX_DIRECTION_REFUTES, "why": a.why,
    })
    print("方向 %s 已证伪 %d/%d —— %s" % (dir_id, refutes, MAX_DIRECTION_REFUTES, a.why))
    if refutes >= MAX_DIRECTION_REFUTES:
        print(_c("方向 %s 已封：loop.py 不再受理 direction=%s 的任何候选。"
                 "下一个候选必须来自 op-constrain 重定后的约束。" % (dir_id, dir_id), WARN))
    else:
        print("回 op-constrain：这条约束不成立，重定一个再开候选。")
    return 0


def cmd_refute(a):
    print(_c("refute 已拆成两条（P0-3）：", WARN))
    print("  impl-broken <tag>        这次实现没写成/没跑对，留在 optimize 修，不耗方向预算")
    print("  direction-refuted <tag>  跑通了但这套攻击面不行，回 constrain 重定约束")
    return 2


def adopted_text(text, tag):
    """把候选条目的 status 改成 adopted（纯文本改写，落盘前由 _write_candidates 校验）。"""
    blk = _cand_block(text, tag)
    if blk:
        seg = text[blk[0]:blk[1]]
        if re.search(r"^    status: .*$", seg, re.M):
            seg = re.sub(r"^    status: .*$", "    status: adopted", seg, count=1, flags=re.M)
        else:
            seg = seg.rstrip("\n") + "\n    status: adopted"
        seg = re.sub(r"^    adopted_on: .*\n", "", seg, flags=re.M)
        seg = seg.rstrip("\n") + "\n    adopted_on: %s\n" % time.strftime("%Y-%m-%d")
        return text[:blk[0]] + seg + text[blk[1]:]
    m = re.search(r"^candidates:[ \t]*$\n?", text, re.M)
    entry = ("  - id: %s\n    status: adopted\n    adopted_on: %s\n"
             % (_scalar(tag), time.strftime("%Y-%m-%d")))
    if not m:
        return text.rstrip("\n") + "\ncandidates:\n" + entry
    return text[:m.end()] + entry + text[m.end():]


def run_done_when(cmd, card):
    """执行候选的 done_when。返回 (rc, 合并后的 stdout+stderr)。

    cwd = 仓库根（`scripts/opcheck.py` 这类共享脚本按仓库根相对路径写才对）；
    卡片目录用环境变量 `OPS_CARD` 暴露给命令，仓库根是 `OPS_REPO_ROOT`。
    超时按失败算（rc=124）——「跑不完」不能读成「通过」。
    """
    env = dict(os.environ)
    env["OPS_CARD"] = str(card)
    env["OPS_REPO_ROOT"] = str(REPO_ROOT)
    try:
        cp = subprocess.run(cmd, shell=True, cwd=str(REPO_ROOT), env=env,
                            capture_output=True, text=True, timeout=VERIFY_TIMEOUT_S)
    except subprocess.TimeoutExpired as e:
        def _s(x):
            if isinstance(x, bytes):
                return x.decode("utf-8", "replace")
            return x or ""
        return 124, ("done_when 超过 %ss 未结束：%s\n%s"
                     % (VERIFY_TIMEOUT_S, cmd, _s(e.stdout) + _s(e.stderr)))
    return cp.returncode, (cp.stdout or "") + (cp.stderr or "")


def cmd_verify(a):
    """完成判定：**只有 done_when 返回 0 才允许 adopted**。

    失败时保持原状态（不改 YAML），只回显尾部 20 行 + 记 `done_when_rc`——
    「我看日志觉得它没问题」不是通过条件。
    """
    card = a.card
    cands = load_candidates(card)
    cand = next((c for c in cands["candidates"]
                 if isinstance(c, dict) and c.get("id") == a.tag), None)
    if cand is None:
        print(_c("拒绝：%s 不在 candidates 段里（没有候选就没有完成判定可谈）" % a.tag, WARN))
        return 2
    cmd = str(cand.get("done_when") or "").strip()
    if not cmd:
        print(_c("拒绝：%s 没有 done_when —— 完成判定不能由模型口述。先在 candidates.yaml "
                 "给它写一条命令（写法见 op-optimize/SKILL.md §七）" % a.tag, WARN))
        return 2

    print("$ " + cmd, flush=True)
    rc, out = run_done_when(cmd, card)
    tail = "\n".join(out.splitlines()[-VERIFY_TAIL_LINES:])
    if rc != 0:
        if tail:
            print(tail)
        print(_c("verify 失败（done_when rc=%d）：%s **保持原状态**（%s）。"
                 "完成判定没通过，不许写 adopted。"
                 % (rc, a.tag, cand.get("status") or "pending"), WARN))
        if str(cand.get("status") or "") == "adopted":
            print(_c("⚠ 它现在是 adopted 但自己的 done_when 不过 —— 这条 adopted 不成立，"
                     "该降级并查为什么之前判过了。", WARN))
        append_record(card, {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "card": card.name, "tag": a.tag,
            "direction": cand.get("direction"), "decision": "VERIFY_FAILED",
            "done_when": cmd, "done_when_rc": rc,
            "status": str(cand.get("status") or "pending"),
        })
        return 1

    cur = str(cand.get("status") or "pending")
    if cur == "verified":
        # verified 是另一个终结态（知识卡/事实条目），复验通过只留记录，不改状态词
        print("%s 已是 verified；done_when 返回 0（复验通过），状态保持 verified" % a.tag)
    else:
        p = card / "candidates.yaml"
        _write_candidates(p, adopted_text(p.read_text(), a.tag))
        cur = "adopted"
        print("%s 的 done_when 返回 0 ⇒ status=adopted（完成判定由命令裁决）" % a.tag)
    append_record(card, {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "card": card.name, "tag": a.tag,
        "direction": cand.get("direction"), "decision": "VERIFIED",
        "done_when": cmd, "done_when_rc": 0, "status": cur,
    })
    return 0


def cmd_adopt(a):
    """旧名，等价于 verify：不跑命令就 adopted 的路已经关掉。"""
    print(_c("adopt 已并入 verify（同一个动作）：完成判定由 done_when 裁决，不由模型口述。", WARN))
    return cmd_verify(a)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-C", "--card", required=True, help="算子卡目录（含 candidates.yaml）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="当前最强 + 待做 + 记录数")
    sub.add_parser("queue", help="打印候选队列与已证伪清单")
    sub.add_parser("report", help="从记录渲染交付表")

    e = sub.add_parser("enumerate", help="列出攻击面（direction 只能从这挑），标出已封的")
    e.add_argument("--new", action="store_true", help="只打印可用的 candidates.yaml 片段")

    r = sub.add_parser("run", help="跑一个实验并判定")
    r.add_argument("tags", nargs="+")
    r.add_argument("--skv", type=int, nargs="+")
    r.add_argument("--s", type=int)
    r.add_argument("--topk", type=int)
    r.add_argument("--nvalid", type=int)
    r.add_argument("--heads", type=int)
    r.add_argument("--qscale", type=float, nargs="+")
    r.add_argument("--rounds", type=int, default=3)
    r.add_argument("--tol", type=float, help="对 fp32 参考的 max_abs 容差（默认读 contract.yaml）")
    r.add_argument("--primary-qscale", type=float, help="主口径 logit 尺度（默认 1；压力档只报退化倍数）")
    r.add_argument("--stress-mult", type=float, help="压力档允许的相对退化倍数（默认 1.5）")
    r.add_argument("--no-acc", action="store_true", help="跳过精度探针（只在你确定逐位一致时用）")
    r.add_argument("--force", action="store_true", help="允许重跑已证伪的变体")

    i = sub.add_parser("impl-broken", help="实现级失败：修实现，不耗方向预算")
    i.add_argument("tag")
    i.add_argument("--why", required=True)

    d = sub.add_parser("direction-refuted", help="方向级失败：回 constrain 重定约束")
    d.add_argument("tag", nargs="?")
    d.add_argument("--direction", help="直接点名方向（候选已不在队列里时用）")
    d.add_argument("--why", required=True)

    f = sub.add_parser("refute", help=argparse.SUPPRESS)   # 已拆分，保留只为给出迁移提示
    f.add_argument("tag", nargs="?")
    f.add_argument("--why")

    v = sub.add_parser("verify", help="跑候选的 done_when；rc=0 才允许 adopted")
    v.add_argument("--tag", required=True)

    a2 = sub.add_parser("adopt", help="旧名，等价于 verify（完成判定由命令裁决）")
    a2.add_argument("tag")

    a = ap.parse_args()
    a.card = Path(a.card).resolve()
    return {"status": cmd_status, "queue": cmd_queue, "run": cmd_run,
            "report": cmd_report, "enumerate": cmd_enumerate,
            "impl-broken": cmd_impl_broken,
            "direction-refuted": cmd_direction_refuted,
            "refute": cmd_refute, "verify": cmd_verify,
            "adopt": cmd_adopt}[a.cmd](a) or 0


if __name__ == "__main__":
    sys.exit(main())
