# -*- coding: utf-8 -*-
"""ledger —— 记账：候选计划、判死/停滞流水、overlay 与进度文件。

## 为什么单独一个模块

`scripts/advance.py` 原来的 1058 行里混着三种东西：
  ① **纯数据结构与纯函数**（Cand/Top、parse_gain、replan、ledger_cites）
  ② 读写运行时文件的记账动作（runs/loop/*.jsonl、overlay、progress.json）
  ③ 编排骨架（run_advance 的 while 轮次、子进程调 loop.py、stop 判据）

①② 与「谁在跑编排」无关：`explore --apply` 也要写 overlay，
`check_diagnosis` 也要读流水，测试也要独立验证 replan 的七个分支。
把它们留在 advance.py 里，等于逼着每个消费者 import 一个有子进程、
有 GPU 假设、有 1000 行的编排器。这里只装 ①②，③ 留在 scripts/advance.py
（原因见该文件 run_advance 的注释：它读模块级 SURFACES_PATH，测试用
monkeypatch 改的就是那个名字，搬走会让补丁静默失效）。

## 一条纪律

本模块**不产生判定**：decision 是 loop.py 写的，这里只读；replan 是编排策略，
它只按 EV 数值和 ledger 里已有的 verdict 做决定，绝不重算显著性。
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

# advance 自己的进度文件（loop.py 写 runs/loop/，advance 写 runs/advance/）。
# **不写卡片**：写范围禁止碰任何 cards，所以 stage 只能落在运行时目录里。
ADVANCE_DIR = "advance"

# overlay 是「自动生成的候选」唯一合法的登记处（loop.py 只读合并它）。
FORGE_OVERLAY = "overlay.yaml"

# 阶段是有序的；「stage 前进」= 索引只增不减（blocked/stopped 是终态另算）。
STAGES = ("idle", "planned", "gated", "ran", "judged", "recorded", "replanned")
TERMINAL = ("blocked", "stopped")


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


# ---------------------------------------------------------------- 读流水

def read_records(card):
    """读 `runs/loop/*.jsonl` —— 只读 loop.py 的产出，不产生任何判定。

    只扫顶层（不进 `noise/`）：噪声记录已被 loop.py 隔离，把它们算进
    「上一轮跑了什么」会让 replan 按一次没形成决策的测量改路线。
    """
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


# ---------------------------------------------------------------- plan

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


def sweep_key(cid):
    """同一批 sweep 的变体共享同一前缀：auto-<template>-h<H>-t<N>-v<M> → 去掉末段。"""
    m = re.match(r"^(auto-[a-z0-9]+-h\d+)-t\d+-v\d+$", str(cid))
    return m.group(1) if m else None


def build_plan(cands, surfaces, ledger=None):
    """候选队列 + surfaces.yaml → top3（可能因 sweep 兄弟而多于 3）。

    排序只按 `expected_gain` 解析出的代理值（简化版，不做完整 EV）。
    只挑 pending 候选：loop.py 的登记门要求候选在队列里，plan 不越权发明候选。

    `cands` / `surfaces` 由调用方传进来（而不是在这里读文件）：
    调用方持有「从哪读」的知识（比如测试会 monkeypatch 路径常量），
    读文件的方式一换，这里不用动。
    """
    surf = {str(s["id"]): s for s in (surfaces or []) if isinstance(s, dict) and s.get("id")}
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
    # top3 是「一次跑几个**不同方向**」的上限，但同一次 sweep 出来的兄弟变体
    # 必须整组一起跑 —— 它们共用一次 base 测量才可比，被截断的那组就白生成了。
    # 判据：id 前缀相同（auto-<template>-h<H>-）即视为同一批 sweep。
    top = items[:3]
    seen = {sweep_key(c.id) for c in top}
    for c in items[3:]:
        k = sweep_key(c.id)
        if k and k in seen:
            top.append(c)
    rejected = RejectedList(items[3:], ledger=ledger)
    return Top(top, rejected=rejected, ledger=ledger)


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
    # prev_top 为空时「掉一半」无从比（与上一轮不可比），按不塌缩处理；
    # 直接跳过该判据而不是抛 IndexError——编排器不能因为数据稀疏就崩。
    if prev_top and new_top[0].ev < 0.5 * prev_top[0].ev:
        return {"action": "STOP", "reason": "EV 掉了一半以上"}
    if prev_top and prev_top[0].id == new_top[0].id:
        return {"action": "CONTINUE_SAME", "reason": "top1 未变"}
    # 只有 1 个候选时不存在「拉开差距」的问题（没有第二名可比）。
    if len(new_top) >= 2 and new_top[0].ev < 1.5 * new_top[1].ev:
        return {"action": "STALLED", "reason": "没拉开差距，可能是抖动"}
    cited = ledger_cites(new_top.rejected, prev_top[0] if prev_top else Cand("?", "", 0.0),
                         ledger)
    return {"action": "CONTINUE_SWITCH", "reason": "换向", "cited": cited}


# ---------------------------------------------------------------- overlay

def write_overlay(card, entries):
    """把生成的候选写进 runs/advance/overlay.yaml（**只新增，不覆盖**）。

    loop.py 的 `load_candidates` 会合并它，并拒绝同 id 冲突。
    这里也做一次同 id 检查：两次生成同名变体时应当停下，而不是让第二次静默赢。
    """
    import sys
    card_io = _card()
    import yaml
    d = Path(card) / "runs" / ADVANCE_DIR
    d.mkdir(parents=True, exist_ok=True)
    f = d / FORGE_OVERLAY
    cur = {"candidates": []}
    if f.is_file():
        cur = card_io.read_yaml(f, {}) or {"candidates": []}
    have = {c.get("id") for c in cur.get("candidates") or [] if isinstance(c, dict)}
    for e in entries:
        if e["id"] in have:
            raise SystemExit(
                "overlay 已存在候选 %r。**不覆盖**：同名变体两次生成会让"
                "「这轮到底测的是哪个实现」无法回答。" % e["id"])
        cur.setdefault("candidates", []).append(e)
        have.add(e["id"])
    f.write_text(yaml.safe_dump(cur, allow_unicode=True, sort_keys=False),
                 encoding="utf-8")
    return f


def overlay_ids(card):
    """overlay 里已登记的候选 id（用于「别再生成一遍」）。"""
    card_io = _card()
    f = Path(card) / "runs" / ADVANCE_DIR / FORGE_OVERLAY
    if not f.is_file():
        return set()
    d = card_io.read_yaml(f, {})
    return {c.get("id") for c in d.get("candidates") or [] if isinstance(c, dict)}


# ---------------------------------------------------------------- progress

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


def advance_stage(state, stage):
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


# ---------------------------------------------------------------- 内部

def _card():
    """按路径加载 card 模块（见 engine/loop.py::_engine 的说明）。"""
    import sys
    key = "_ops_engine_card"
    if key in sys.modules:
        return sys.modules[key]
    import importlib.util
    spec = importlib.util.spec_from_file_location(key, str(Path(__file__).resolve().parent / "card.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod
