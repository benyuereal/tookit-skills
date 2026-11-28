#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""消融流水线：**只编排，不判定**。

归因判据只有一份实现（`scripts/ablate.py` + `loop.py::_significant`）。本模块
只补它前面缺的那一环：

    preflight_ablate   开跑前的三条判据（不满足就 fail fast）
    ablate_prompt      给 agent 的 prompt（模板在 instrument/prompts/ablate.md）
    ablation_tags      `<champion>-no-<method>` 的 tag 列表
    build_spec         从 run 记录生成 ablate.py 的 spec
    run_ablation       逐个 `backend.eval(tag)` 并收集记录（不调 agent）

**不调用 `ablate.py` 本体** —— 那一步（子进程 + 退出码 → verdict）留给调用方，
免得这里长出第二套判据。**只读**：不写 runs/、不改卡片。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ENGINE_DIR = Path(__file__).resolve().parent
if str(_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(_ENGINE_DIR))

from instrument import render  # noqa: E402

_ID_KEYS = ("id", "name", "method", "title")
_DESC_KEYS = ("desc", "description", "statement", "hypothesis", "summary")


# --------------------------------------------------------------------- 记录读取
def _load_yaml(path: Path) -> dict:
    try:
        import yaml
        d = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, ValueError, ImportError):
        return {}
    return d if isinstance(d, dict) else {}


def _repo_root(card: Path) -> Path | None:
    """卡片所在仓库根（`.artifacts` 的锚点）。找不到就退到卡片父目录。"""
    for p in (card, *card.parents):
        if (p / ".git").exists() or (p / "Plugins" / "ops").is_dir() \
                or (p / "scripts" / "ablate.py").is_file():
            return p
    return card.parent if card.parent != card else None


def _artifact_roots(card: Path) -> list[Path]:
    """可能要搜的 `.artifacts` 目录（去重，保持从近到远）。"""
    roots: list[Path] = []
    env = os.environ.get("OPS_ARTIFACTS")
    if env:
        roots.append(Path(env))
    for p in (card, *card.parents):
        a = p / ".artifacts"
        if a.is_dir():
            roots.append(a)
    out, seen = [], set()
    for r in roots:
        try:
            k = str(r.resolve())
        except OSError:
            k = str(r)
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def _record_files(card: Path):
    """产出 `(file, strict)` —— strict 的文件要按记录里的 card 名过滤。"""
    seen = set()

    def emit(f: Path, strict: bool):
        try:
            k = str(f.resolve())
        except OSError:
            k = str(f)
        if k in seen:
            return
        seen.add(k)
        yield f, strict

    legacy = card / "runs" / "loop"
    if legacy.is_dir():
        for f in sorted(legacy.glob("*.jsonl")):
            yield from emit(f, False)
    for root in _artifact_roots(card):
        for f in sorted(root.glob("**/runs/loop/*.jsonl")):
            yield from emit(f, True)


def read_records(card: Path) -> dict[str, dict]:
    """tag → 最新一条记录（只读）。同一 tag 多条时按 `ts` 取最新。

    与 `scripts/ablate.py::read_records` 同一口径：只读 `runs/loop/*.jsonl`
    顶层，`noise/` 隔离区不读。
    """
    card = Path(card)
    best: dict[str, tuple[tuple[str, int], dict]] = {}
    seq = 0
    for f, strict in _record_files(card):
        try:
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            seq += 1
            if not isinstance(r, dict) or not r.get("tag"):
                continue
            if strict and r.get("card") and r["card"] != card.name:
                continue
            key = (str(r.get("ts") or ""), seq)
            tag = str(r["tag"])
            if tag not in best or key >= best[tag][0]:
                best[tag] = (key, r)
    return {tag: rec for tag, (_k, rec) in best.items()}


def latest_record(card: Path, tag: str) -> dict | None:
    return read_records(card).get(str(tag))


# --------------------------------------------------------------------- methods
def _method_id(m) -> str:
    if isinstance(m, str):
        return m.strip()
    if isinstance(m, dict):
        for k in _ID_KEYS:
            v = m.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return ""


def _method_desc(m) -> str:
    if isinstance(m, dict):
        for k in _DESC_KEYS:
            v = m.get(k)
            if isinstance(v, str) and v.strip():
                return " ".join(v.split())
    return ""


def normalize_methods(methods) -> list[dict]:
    """把 `methods:` 段规范成 `[{id, desc, raw}]`；没有 id 的条目丢弃（fail-closed）。"""
    out = []
    for m in methods or []:
        mid = _method_id(m)
        if not mid:
            continue
        out.append({"id": mid, "desc": _method_desc(m), "raw": m})
    return out


def load_methods(card: Path) -> list[dict]:
    """读 `<card>/candidates.yaml` 的 `methods:` 段（只读，不猜默认值）。"""
    d = _load_yaml(Path(card) / "candidates.yaml")
    return normalize_methods(d.get("methods"))


# --------------------------------------------------------------------- 前置条件
def preflight_ablate(card: Path, champion_tag: str) -> list[str]:
    """返回问题列表；非空拒绝开跑（计划 §2.1 的三条判据）。"""
    probs: list[str] = []
    rec = latest_record(Path(card), champion_tag)
    if not rec:
        probs.append("冠军 tag %r 没有 run 记录 —— 先跑一次再消融" % champion_tag)
    elif str(rec.get("measure_mode") or "") != "event_isolated":
        probs.append("冠军记录不是 event_isolated，不可 claim，消融没有意义")
    methods = load_methods(Path(card))
    if len(methods) < 2:
        probs.append("方法数 < 2（当前 %d）—— 消融至少要能删两项" % len(methods))
    return probs


# --------------------------------------------------------------------- prompt
def ablation_tags(champion_tag: str, methods) -> list[str]:
    """`<champion>-no-<method>`；顺序与 methods 一致，去重保序。"""
    out, seen = [], set()
    for m in normalize_methods(methods):
        tag = "%s-no-%s" % (champion_tag, m["id"])
        if tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


def ablate_prompt(card, champion_tag: str, methods) -> str:
    """给 agent 的消融 prompt；文本在 `instrument/prompts/ablate.md`。"""
    ms = normalize_methods(methods)
    card_name = Path(card).name if card else ""
    method_lines = []
    for m in ms:
        method_lines.append("- `%s`%s" % (m["id"], ("：" + m["desc"]) if m["desc"] else ""))
    tag_lines = ["- `%s`" % t for t in ablation_tags(champion_tag, ms)]
    return render(
        "ablate",
        card_name=card_name,
        card_suffix=("：" + card_name) if card_name else "",
        champion_tag=champion_tag,
        method_count=len(ms),
        methods_block="\n".join(method_lines) or "（candidates.yaml 里没有 methods）",
        tag_lines="\n".join(tag_lines) or "（无）",
    )


# --------------------------------------------------------------------- spec
def _num(rec: dict | None, key: str) -> float | None:
    v = (rec or {}).get(key)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return None


def _records_map(records) -> dict[str, dict]:
    """接受 `{tag: rec}` 或 `[rec, ...]`（列表项必须带 `tag`）。"""
    if isinstance(records, dict):
        out = {}
        for k, v in records.items():
            if not isinstance(v, dict):
                continue
            r = dict(v)
            r.setdefault("tag", str(k))
            out[str(k)] = r
        return out
    out = {}
    for r in records or []:
        if isinstance(r, dict) and r.get("tag"):
            out[str(r["tag"])] = r
    return out


def _numerically_valid(rec: dict | None) -> bool:
    """记录里的数值正确性；缺证据 → False（fail-closed，不许当 true）。"""
    if not rec:
        return False
    if isinstance(rec.get("acc_ok"), bool):
        return rec["acc_ok"]
    if isinstance(rec.get("correctness"), bool):
        return rec["correctness"]
    return False


def _baseline_tag(champion: dict | None, recs: dict[str, dict],
                  override: str | None) -> str:
    if override:
        return str(override)
    for k in ("base_tag", "baseline_tag"):
        v = (champion or {}).get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    for t in ("base", "baseline", "ref-heavy", "ref"):
        if t in recs:
            return t
    return "base"


def build_spec(champion_tag: str, methods, records,
               baseline_tag: str | None = None) -> dict:
    """按计划 §2.3 生成 ablate.py 的 spec。

    `share` 是**毛比值** `cand_ms / champion_ms`（诊断用）——不是归因份额。
    真正的 `delta/gain` 由 `scripts/ablate.py` 单一实现计算，这里不复制判据。
    """
    ms = normalize_methods(methods)
    recs = _records_map(records)
    champion = recs.get(str(champion_tag))
    champ_ms = _num(champion, "cand_ms")
    ablations = []
    for m in ms:
        tag = "%s-no-%s" % (champion_tag, m["id"])
        rec = recs.get(tag)
        cand_ms = _num(rec, "cand_ms")
        share = (cand_ms / champ_ms) if (cand_ms is not None and champ_ms) else None
        delta = (cand_ms - champ_ms) if (cand_ms is not None and champ_ms is not None) else None
        ablations.append({
            "id": m["id"],
            "tag": tag,
            "derived_from": "champion",
            "numerically_valid": _numerically_valid(rec),
            "share": share,
            "delta_ms": delta,
        })
    return {
        "baseline": _baseline_tag(champion, recs, baseline_tag),
        "champion": str(champion_tag),
        "ablations": ablations,
    }


# --------------------------------------------------------------------- 编排
def run_ablation(backend, card, champion_tag: str, methods) -> dict:
    """逐个 `backend.eval(tag)`，**不启动 agent**；返回 {tags, spec, records}。

    不调用 `ablate.py` 本体 —— 归因那一步交给调用方（子进程 + 退出码）。
    """
    card = Path(card)
    ms = normalize_methods(methods)
    tags = ablation_tags(champion_tag, ms)
    records: dict[str, dict] = {}
    for tag in tags:
        r = backend.eval(tag) or {}
        r = dict(r)
        r.setdefault("tag", tag)
        records[tag] = r
    # spec 的 share 需要冠军的 cand_ms；冠军记录已在盘上时只读取用
    merged = dict(records)
    champ = latest_record(card, champion_tag)
    if champ:
        merged[str(champion_tag)] = champ
    return {
        "tags": tags,
        "spec": build_spec(champion_tag, ms, merged),
        "records": records,
    }


__all__ = [
    "read_records", "latest_record", "load_methods", "normalize_methods",
    "preflight_ablate", "ablate_prompt", "ablation_tags", "build_spec",
    "run_ablation",
]
