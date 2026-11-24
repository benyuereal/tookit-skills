#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""advance —— 多轮编排的**唯一入口**（v5.1 §6.2）。

它只做一件事：把「计划 → 前提门 → 跑实验 → 判 → 记 → 重规划 → 停」串起来，
**不重实现任何判定**。

铁律（违反即为安全性倒退）
--------------------------
`loop.py run` 已有六道门：
  ① candidates 登记  ② bottleneck 归因  ③ done_when 外部裁决
  ④ 方向预算  ⑤ refuted 拒跑  ⑥ allowed_actions 天花板约束
本模块自己跑实验，这六道门全绕过。所以：

  * run 只通过子进程调 `python3 -I <engine>/loop.py -C <card> run <tag>`
  * judge 只读 loop.py 写进 `runs/loop/*.jsonl` 的 `decision`，不自己算显著性
  * 任何要执行的命令里出现 `ab.py` / `bench/` 一律拒绝（`_guard_no_bench`）

唯一验收
--------
`advance.py --card X --max-rounds 1` 跑完一轮，中间没人敲命令，
`runs/loop/*.jsonl` 多一条记录，且 advance 自己的 `progress.stage` 前进。

    每轮：plan → gate(precondition) → run(调 loop.py) → judge → record
          → replan → stop
"""
import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# scripts/advance.py → 向上找到 .git 即为仓库根（与 loop.py 同一策略）
REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / ".git").exists()),
                 Path(__file__).resolve().parents[1])
ENGINE_LOOP = REPO_ROOT / "Plugins/ops/engine/loop.py"
SURFACES_PATH = REPO_ROOT / "Plugins/ops/skills/op-optimize/references/surfaces.yaml"

# advance 自己的进度文件（loop.py 写 runs/loop/，advance 写 runs/advance/）。
# **不写卡片**：写范围禁止碰任何 cards，所以 stage 只能落在运行时目录里。
ADVANCE_DIR = "advance"
# 阶段是有序的；「stage 前进」= 索引只增不减（blocked/stopped 是终态另算）。
STAGES = ("idle", "planned", "gated", "ran", "judged", "recorded", "replanned")
TERMINAL = ("blocked", "stopped")

# floor.py 的 at_floor 阈值（remaining < 0.10 ⇒ 别再折腾）
AT_FLOOR = 0.10
# 连续 refuted 多少次就停（「连续」只看本轮 decision，不看噪声）
MAX_CONSECUTIVE_REFUTED = 2


def _c(s, color=None):
    return s


# ---------------------------------------------------------------- 数据结构

@dataclass
class Cand:
    """一个候选（来自 candidates.yaml + surfaces.yaml 的排序附加信息）。"""
    id: str
    direction: str
    ev: float
    priority: int = 99
    surface: dict = None
    precondition_check: str = None
    variant: str = None
    raw: dict = None


class RejectedList(list):
    """rejected 列表。带上 ledger，使 `ledger_cites(rejected, prev)` 保持 2 参签名。"""

    def __init__(self, items=(), ledger=None):
        super().__init__(items)
        self.ledger = ledger


class Top(list):
    """top-N 计划。保留 `.rejected` / `.ledger`，与 §6.3 的伪代码对齐。"""

    def __init__(self, items=(), rejected=None, ledger=None):
        super().__init__(items)
        if isinstance(rejected, RejectedList):
            self.rejected = rejected
        else:
            self.rejected = RejectedList(rejected or [], ledger=ledger)
        self.ledger = ledger


# ---------------------------------------------------------------- 知识读取

def _yaml(path):
    try:
        import yaml
        return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, ImportError, ValueError):
        return {}


def load_surfaces():
    d = _yaml(SURFACES_PATH)
    return [s for s in d if isinstance(s, dict) and s.get("id")] if isinstance(d, list) else []


def parse_gain(text):
    """把 surfaces.yaml 的 `expected_gain` 文本解析成一个 0..1 的**代理值**。

    这是 plan 的**简化版**（完整 EV 归 plan.py，后续阶段）。它不是 EV：
    只用于排序与 replan 的塌缩判据。取值口径写死如下，避免「凭感觉」：
      20~60%  → 区间中点 0.40
      ≤15%    → 上界 0.15
      ≈0      → 0.0
      约 2×   → 1.0（2 倍加速 = 收益 100%）
    """
    if not text:
        return 0.0
    s = str(text)
    m = re.search(r"(\d+(?:\.\d+)?)\s*[~～\-]\s*(\d+(?:\.\d+)?)\s*%", s)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2.0 / 100.0
    m = re.search(r"约\s*(\d+(?:\.\d+)?)\s*[×x]", s)
    if m:
        return max(0.0, float(m.group(1)) - 1.0)
    m = re.search(r"[≤<=]\s*(\d+(?:\.\d+)?)\s*%", s)
    if m:
        return float(m.group(1)) / 100.0
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", s)
    if m:
        return float(m.group(1)) / 100.0
    return 0.0


def read_roofline(card):
    p = Path(card) / "roofline.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def card_id(card, roof=None):
    roof = roof if roof is not None else read_roofline(card)
    return str(roof.get("card") or Path(card).name)


def read_records(card):
    """读 `runs/loop/*.jsonl` —— 只读取 loop.py 的产出，不产生任何判定。"""
    out = []
    d = Path(card) / "runs" / "loop"
    if not d.is_dir():
        return out
    for f in sorted(d.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def _rel(p):
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except (ValueError, OSError):
        return str(Path(p).resolve())


def pick_results_json(card):
    """给 `<results.json>` 占位符挑一个真实存在的 results 文件；没有返回 None。"""
    d = Path(card) / "results"
    preferred = d / "impl-ranking.json"
    if preferred.is_file():
        return _rel(preferred)
    if d.is_dir():
        for p in sorted(d.glob("*.json")):
            return _rel(p)
    return None


# ---------------------------------------------------------------- plan（简化版）

def build_plan(card, ledger=None):
    """读 candidates.yaml + surfaces.yaml，给出 top3。

    排序只按 `expected_gain` 解析出的代理值（简化版，不做完整 EV）。
    只挑 pending 候选：loop.py 的登记门要求候选在队列里，plan 不越权发明候选。
    """
    cands = _yaml(Path(card) / "candidates.yaml")
    surf = {str(s["id"]): s for s in load_surfaces()}
    items = []
    for c in cands.get("candidates") or []:
        if not isinstance(c, dict):
            continue
        if str(c.get("status") or "pending") != "pending":
            continue
        d = str(c.get("direction") or "")
        s = surf.get(d) or {}
        if isinstance(c.get("ev"), (int, float)):
            ev = float(c["ev"])
        else:
            ev = parse_gain(s.get("expected_gain"))
        items.append(Cand(
            id=str(c.get("id") or "?"), direction=d, ev=ev,
            priority=int(c.get("priority") or 99), surface=s,
            precondition_check=s.get("precondition_check"),
            variant=c.get("variant"), raw=c))
    items.sort(key=lambda x: (-x.ev, x.priority, x.id))
    top = items[:3]
    rejected = RejectedList(items[3:], ledger=ledger)
    return Top(top, rejected=rejected, ledger=ledger)


# ---------------------------------------------------------------- gate（前提门）

def _subst_check(cmd, card, cid, results_rel):
    s = str(cmd).replace("<card>", cid)
    if "<results.json>" in s:
        if not results_rel:
            return None                      # 占位符无法落地 = unrunnable
        s = s.replace("<results.json>", results_rel)
    return s


def gate(cands, card, runner):
    """跑候选的 precondition_check。返回 (通过的候选列表, 逐条日志)。

    **不跑实验**：precondition_check 是 roofline/PMC/ISA 的读数命令，不是 bench。
    任一命令含 ab.py / bench/ 直接拒绝（见 _guard_no_bench）。
    """
    cid = card_id(card)
    results_rel = pick_results_json(card)
    passed, log = [], []
    for c in cands:
        if not c.precondition_check:
            passed.append(c)
            log.append({"id": c.id, "check": None, "ok": True, "why": "无 precondition_check"})
            continue
        cmd = _subst_check(c.precondition_check, card, cid, results_rel)
        if cmd is None:
            log.append({"id": c.id, "check": c.precondition_check, "ok": False,
                        "why": "unrunnable：占位符无法落地"})
            continue
        rc, out = runner.precondition(cmd, cwd=REPO_ROOT)
        ok = (rc == 0)
        log.append({"id": c.id, "check": cmd, "ok": ok, "rc": rc,
                    "why": "" if ok else "precondition 未通过 (rc=%d)" % rc})
        if ok:
            passed.append(c)
    return passed, log


# ---------------------------------------------------------------- run（只调 loop.py）

def _guard_no_bench(cmd):
    """硬防护：advance 绝不直接跑卡片的 bench。

    唯一的实验入口是 loop.py；出现 `ab.py` 或 `bench/` 就说明有人想绕过六道门。
    """
    s = " ".join(cmd) if isinstance(cmd, (list, tuple)) else str(cmd)
    low = s.lower()
    if "ab.py" in low or "/bench/" in low or "bench/ab" in low:
        raise SystemExit("advance 拒绝直接跑 bench（%s）：实验只能经 loop.py run" % s)


def loop_run_cmd(card, tag):
    return [sys.executable, "-I", str(ENGINE_LOOP), "-C", str(card), "run", str(tag)]


class Runner:
    """默认执行器。测试可替换这些方法，而无需真 GPU。"""

    def __init__(self, dry_run=False, timeout=600):
        self.dry_run = dry_run
        self.timeout = timeout
        self.commands = []          # 审计：记录所有实际执行的命令

    def precondition(self, cmd, cwd):
        _guard_no_bench(cmd)
        self.commands.append(cmd)
        try:
            cp = subprocess.run(cmd, shell=True, cwd=str(cwd), capture_output=True,
                                text=True, timeout=self.timeout)
            return cp.returncode, (cp.stdout or "") + (cp.stderr or "")
        except (OSError, subprocess.SubprocessError) as e:      # noqa: BLE001
            return 1, str(e)

    def loop_run(self, card, tag):
        cmd = loop_run_cmd(card, tag)
        _guard_no_bench(cmd)
        self.commands.append(" ".join(cmd))
        cp = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True,
                            timeout=self.timeout)
        return cp.returncode, (cp.stdout or "") + (cp.stderr or "")


# ---------------------------------------------------------------- judge / record

def judge(card, before_len):
    """从 loop.py 的产出里读 decision。**不自己算显著性、不重判。**

    loop.py 在一个 tag 下会为每个 variant×shape 各写一条记录；这里取本轮新增
    记录的最后一条为主判定，并原样保留全部记录供报告。
    """
    recs = read_records(card)
    new = recs[before_len:]
    if not new:
        return {"decision": None, "records": [], "why": "本轮没有新增记录"}
    return {"decision": new[-1].get("decision"), "records": new, "why": "读自 runs/loop"}


def confirm_record(card, before_len):
    """确认 loop.py 的 append 真的落盘了（advance 只确认，不代写）。"""
    recs = read_records(card)
    new = recs[before_len:]
    return len(new) > 0, new


# ---------------------------------------------------------------- replan（v5.1 §6.3）

def ledger_cites(rejected, prev, ledger=None):
    """新 top1 的取舍必须引用上一轮 verdict。返回 citation 或 None。

    两个来源：
      1. rejected 条目显式 `cites` 了 prev.id（候选自己写的理由）
      2. ledger（runs/loop 记录）里存在 prev.id 的 verdict（机器记录）
    """
    pid = getattr(prev, "id", None) or (prev.get("id") if isinstance(prev, dict) else None)
    if not pid:
        return None
    ledger = ledger if ledger is not None else getattr(rejected, "ledger", None)

    for item in rejected or []:
        raw = getattr(item, "raw", None) or {}
        cites = raw.get("cites") or []
        if pid in [str(x) for x in cites]:
            return {"tag": pid, "verdict": "cited-by-candidate",
                    "card": None, "ts": None}

    for r in ledger or []:
        if r.get("tag") == pid or r.get("candidate") == pid:
            verdict = r.get("verdict") or r.get("decision")
            if verdict:
                return {"card": r.get("card"), "ts": r.get("ts"),
                        "tag": pid, "verdict": verdict}
    return None


def replan(prev_top, new_top, ledger=None):
    """§6.3 防 EV 塌缩。**逐条照实现**，分支顺序不能改。"""
    if not new_top:
        return {"action": "STOP", "reason": "无可挖方向"}
    if new_top[0].ev < 0.30:
        return {"action": "STOP", "reason": "EV 塌缩到 0.3 以下"}
    if new_top[0].ev < 0.5 * prev_top[0].ev:
        return {"action": "STOP", "reason": "EV 掉了一半以上"}
    if prev_top[0].id == new_top[0].id:
        return {"action": "CONTINUE_SAME", "reason": "top1 未变"}
    if new_top[0].ev < 1.5 * new_top[1].ev:
        return {"action": "STALLED", "reason": "没拉开差距，可能是抖动"}
    cited = ledger_cites(new_top.rejected, prev_top[0], ledger)
    return {"action": "CONTINUE_SWITCH", "reason": "换向", "cited": cited}


# ---------------------------------------------------------------- progress 状态

def progress_path(card):
    return Path(card) / "runs" / ADVANCE_DIR / "progress.json"


def load_progress(card):
    p = progress_path(card)
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(d, dict):
            return d
    except (OSError, ValueError):
        pass
    return {"card": Path(card).name, "round": 0, "stage": "idle",
            "since": time.strftime("%Y-%m-%dT%H:%M:%S"), "history": []}


def save_progress(card, state):
    p = progress_path(card)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _advance_stage(state, stage):
    """stage 只许前进（终态除外）。返回是否真的前进。"""
    cur = state.get("stage") or "idle"
    if cur in TERMINAL:
        return False
    if stage in TERMINAL:
        state["stage"] = stage
        return True
    if cur in STAGES and stage in STAGES and STAGES.index(stage) > STAGES.index(cur):
        state["stage"] = stage
        return True
    return False


# ---------------------------------------------------------------- 编排

def run_advance(card, max_rounds, runner=None, dry_run=False, emit=print,
                write_state=None):
    """跑多轮。返回报告 dict。dry-run 跳过 run，且不落任何状态。"""
    card = Path(card)
    if write_state is None:
        write_state = not dry_run
    runner = runner or Runner(dry_run=dry_run)
    report = {"card": card_id(card), "dry_run": bool(dry_run), "rounds": 0,
              "status": "done", "reason": "", "gate": [], "decisions": [],
              "replan": None, "progress_stage": "idle", "commands": runner.commands}

    if max_rounds <= 0:
        emit("max-rounds=0：立即退出，不跑任何东西。")
        report.update(status="noop", reason="max-rounds=0", progress_stage="idle")
        return report

    roof = read_roofline(card)
    remaining = roof.get("remaining")
    if isinstance(remaining, (int, float)) and remaining < AT_FLOOR:
        emit("at_floor：remaining=%.4f < %.2f，停在计划前。" % (remaining, AT_FLOOR))
        report.update(status="stopped", reason="at_floor", progress_stage="idle")
        return report

    state = load_progress(card)
    state["card"] = report["card"]
    state["since"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _advance_stage(state, "planned")

    ledger = read_records(card)
    prev_top = build_plan(card, ledger=ledger)
    consecutive_refuted = 0

    def _emit_plan(top):
        for i, c in enumerate(top, 1):
            emit("  top%d  %-22s direction=%-18s ev=%.3f  gain=%r"
                 % (i, c.id, c.direction, c.ev, (c.surface or {}).get("expected_gain")))

    emit("plan（简化版：按 expected_gain 排序，不做完整 EV）")
    _emit_plan(prev_top)
    if not prev_top:
        emit("  没有 pending 候选 → blocked")
        state["stage"] = "blocked"
        if write_state:
            save_progress(card, state)
        report.update(status="blocked", reason="没有 pending 候选",
                      progress_stage=state["stage"])
        return report

    while report["rounds"] < max_rounds:
        report["rounds"] += 1
        ledger = read_records(card)
        plan = build_plan(card, ledger=ledger)
        _advance_stage(state, "planned")

        # ---- gate
        passed, glog = gate(list(plan), card, runner)
        report["gate"] = glog
        _advance_stage(state, "gated")
        emit("\nround %d gate：" % report["rounds"])
        for g in glog:
            emit("  %-22s %s %s" % (g["id"], "✓" if g["ok"] else "✗", g.get("why", "")))
        if not passed:
            emit("  全部 precondition 未通过 → blocked（不进 run）")
            state["stage"] = "blocked"
            if write_state:
                save_progress(card, state)
            report.update(status="blocked", reason="precondition 全部失败",
                          progress_stage=state["stage"])
            return report

        # ---- run（只调 loop.py；被它自己的门拒了就顺延下一个）
        before = len(read_records(card))
        chosen = None
        run_rc = None
        if dry_run:
            chosen = passed[0]
            emit("\nround %d run：[dry-run] 跳过，不调 loop.py（将跑 %s）"
                 % (report["rounds"], chosen.id))
        else:
            for cand in passed:
                rc, out = runner.loop_run(card, cand.id)
                run_rc = rc
                if rc == 0:
                    chosen = cand
                    break
                # rc==2 是 loop.py 的门拒绝（登记/归因/预算/refuted/天花板）——
                # advance 不重判，只顺延下一个候选。其他 rc 视为运行失败。
                emit("  loop.py run %s → rc=%d，顺延" % (cand.id, rc))
            if chosen is None:
                state["stage"] = "blocked"
                if write_state:
                    save_progress(card, state)
                report.update(
                    status="blocked",
                    reason="loop.py 拒绝了全部候选（rc=%r）：六道门生效，未被绕过"
                           % run_rc,
                    progress_stage=state["stage"])
                return report
            _advance_stage(state, "ran")

        # ---- judge（只读 decision）
        if dry_run:
            j = judge(card, before)
            decision = j["decision"] or "DRY_RUN"
            emit("round %d judge：[dry-run] %s（未新增记录时不编造判定）"
                 % (report["rounds"], decision))
        else:
            j = judge(card, before)
            decision = j["decision"]
            emit("\nround %d judge：decision=%s（读自 loop.py 产出）"
                 % (report["rounds"], decision))
        _advance_stage(state, "judged")

        # ---- record（确认，不代写）
        if dry_run:
            emit("round %d record：[dry-run] 不写记录" % report["rounds"])
        else:
            ok, new = confirm_record(card, before)
            emit("round %d record：%s（新增 %d 条）"
                 % (report["rounds"], "已落盘" if ok else "未落盘", len(new)))
            _advance_stage(state, "recorded")
        report["decisions"].append(decision)

        # ---- 连续 refuted 计数
        if decision in ("REJECTED_NO_GAIN", "REJECTED_ACCURACY"):
            consecutive_refuted += 1
        else:
            consecutive_refuted = 0

        # ---- replan
        new_top = build_plan(card, ledger=read_records(card))
        action = replan(plan, new_top, ledger=read_records(card))
        report["replan"] = action
        _advance_stage(state, "replanned")
        emit("round %d replan：%s —— %s%s"
             % (report["rounds"], action["action"], action["reason"],
                "" if action["action"] != "CONTINUE_SWITCH"
                else "（citation=%s）" % (action.get("cited") or "无")))
        state.setdefault("history", []).append(
            {"round": report["rounds"], "decision": decision,
             "replan": action["action"], "ts": time.strftime("%Y-%m-%dT%H:%M:%S")})
        state["round"] = report["rounds"]

        # ---- stop
        if consecutive_refuted >= MAX_CONSECUTIVE_REFUTED:
            emit("stop：连续 %d 轮 refuted。" % consecutive_refuted)
            report.update(status="stopped", reason="连续 refuted")
            break
        if report["rounds"] >= max_rounds:
            emit("stop：达到预算（max-rounds=%d）。" % max_rounds)
            report.update(status="done", reason="budget")
            break
        if action["action"] == "STOP":
            emit("stop：replan 判停 —— %s" % action["reason"])
            report.update(status="stopped", reason=action["reason"])
            break
        if action["action"] == "STALLED":
            # 抖动：不换向，保持 prev_top 再跑一轮
            emit("stalled：没拉开差距，本轮不换向，沿用当前方向。")
        elif action["action"] == "CONTINUE_SWITCH":
            prev_top = new_top
        else:                                  # CONTINUE_SAME
            prev_top = new_top

    report["progress_stage"] = state.get("stage")
    if write_state:
        save_progress(card, state)
    return report


# ---------------------------------------------------------------- selftest

def selftest():
    """不依赖 GPU / 卡片的纯逻辑自检。"""
    def C(i, ev):
        return Cand(id=i, direction=i, ev=ev)

    T = lambda *xs: Top(list(xs))
    cases = []
    cases.append(("replan 无方向 → STOP",
                  replan(T(C("a", 0.8)), T())["action"] == "STOP"))
    cases.append(("EV<0.3 → STOP",
                  replan(T(C("a", 0.8)), T(C("b", 0.2)))["action"] == "STOP"))
    cases.append(("EV 掉一半 → STOP",
                  replan(T(C("a", 0.8)), T(C("b", 0.35)))["action"] == "STOP"))
    cases.append(("top1 未变 → CONTINUE_SAME",
                  replan(T(C("a", 0.5)), T(C("a", 0.5)))["action"] == "CONTINUE_SAME"))
    cases.append(("未领先 1.5× → STALLED",
                  replan(T(C("a", 0.5)), T(C("b", 0.5), C("c", 0.4)))["action"] == "STALLED"))
    sw = replan(T(C("a", 0.5)), T(C("b", 0.8), C("c", 0.2)))
    cases.append(("正常换向 → CONTINUE_SWITCH", sw["action"] == "CONTINUE_SWITCH"))
    cases.append(("parse_gain 区间中点", abs(parse_gain("20~60%（x）") - 0.40) < 1e-9))
    cases.append(("parse_gain 上界", abs(parse_gain("≤15%（x）") - 0.15) < 1e-9))
    banned = False
    try:
        _guard_no_bench("python3 bench/ab.py --json x")
    except SystemExit:
        banned = True
    cases.append(("ab.py 被拦", banned))
    n = 0
    for name, ok in cases:
        print("[%s] %s" % ("ok" if ok else "FAIL", name))
        n += 1 if ok else 0
    print("\nselftest %s（%d/%d）" % ("PASS" if n == len(cases) else "FAIL", n, len(cases)))
    return 0 if n == len(cases) else 1


# ---------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description="多轮编排唯一入口（只编排，不重实现判定）")
    ap.add_argument("--card", help="算子卡目录（含 candidates.yaml / roofline.json）")
    ap.add_argument("--max-rounds", type=int, default=3, dest="max_rounds")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="跳过 run（无需 GPU），只验证编排逻辑")
    ap.add_argument("--json", action="store_true", help="机器可读报告")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()
    if not a.card:
        ap.error("--card 必填（或用 --selftest）")

    report = run_advance(Path(a.card).resolve(), a.max_rounds, dry_run=a.dry_run)
    if a.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("\n== advance 结束 ==")
        print("status   %s（%s）" % (report["status"], report.get("reason") or "-"))
        print("rounds   %d" % report["rounds"])
        print("stage    %s" % report.get("progress_stage"))
        print("decisions %s" % (report.get("decisions") or "-"))
    return 0 if report["status"] in ("done", "noop", "stopped") else 2


if __name__ == "__main__":
    sys.exit(main())
