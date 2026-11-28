#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""迭代 driver：控制流 + CLI。

分层（一个模块一件事）：
    contract.py   IterationRecord / Backend / 判定与噪声策略
    prompt.py     给 agent 的结构化 prompt
    backends.py   FakeBackend / 真 backend（git + agent + loop.py）
    driver.py     iterate() 控制流 + CLI（本文件）

真正的测量与判定在 loop.py；本文件只决定「什么时候调谁、下一轮喂什么」。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# 直接执行本文件时，同目录的 contract/prompt/backends 需要可 import
sys.path.insert(0, str(Path(__file__).resolve().parent))

from contract import (  # noqa: E402
    Backend, IterationRecord, MAX_NOOP, MAX_PLATEAU, NOISE_FLOOR,
    agent_changes, decide_verdict, noise_floor_from,
)
from prompt import render_student_prompt  # noqa: E402
from backends import FakeBackend, RealBackend  # noqa: E402,F401  (re-export)

__all__ = [
    "Backend", "IterationRecord", "FakeBackend", "RealBackend",
    "agent_changes", "decide_verdict", "noise_floor_from",
    "render_student_prompt", "iterate", "repo_records_dir", "main",
]


def repo_records_dir(ws) -> Path:
    """记录默认落到**仓库级** .artifacts/driver/records/（R4）。"""
    env = os.environ.get("OPS_RECORDS_DIR")
    if env:
        return Path(env)
    p = Path(ws).resolve()
    for q in [p, *p.parents]:
        if (q / "Plugins/ops/engine/loop.py").is_file() and ".artifacts" not in q.parts:
            return q / ".artifacts" / "driver" / "records"
    return Path(__file__).resolve().parents[3] / ".artifacts" / "driver" / "records"


def iterate(backend: Backend, rounds: int, card: str = "",
            noise_floor: float = NOISE_FLOOR, max_plateau: int = MAX_PLATEAU,
            max_impl_fails: int | None = None,
            candidate: dict | None = None,
            on_round=None) -> list[IterationRecord]:
    """跑 N 轮。**纯控制流**：所有副作用都经 backend。"""
    if max_impl_fails is None:
        try:
            from loop import MAX_IMPL_FAILS as max_impl_fails
        except Exception:                                    # noqa: BLE001
            max_impl_fails = 3
    records: list[IterationRecord] = []
    ratios: list[float] = []
    head = backend.commit("baseline")
    plateau = 0
    impl_fails = 0
    noops = 0
    green = False

    for i in range(1, rounds + 1):
        prompt = render_student_prompt(records, card, candidate)
        change = backend.run_agent(prompt, role="student")
        commit = backend.commit(change or ("round %d" % i))
        files = list(getattr(backend, "changed_files", []) or [])
        noop = not agent_changes(files)
        if noop:
            diag = []
            if hasattr(backend, "scratch_files"):
                sf = backend.scratch_files()
                if sf:
                    diag.append("scratch=%s" % ",".join(sf[:5]))
            if diag:
                change = (change + " | " + " ".join(diag))[:600]
        if hasattr(backend, "no_promote"):
            backend.no_promote = bool(noop)
        ev = backend.eval(("r%d" % i))

        correctness = ev.get("correctness")
        if correctness is True:
            green = True
        base_ms, cand_ms = ev.get("base_ms"), ev.get("cand_ms")
        delta = (cand_ms / base_ms - 1.0) if (base_ms and cand_ms) else None

        # R1：噪声底 = 同一次 job 重复测量的 spread（loop.py --repeat），下限 1.5%
        _rep = ev.get("noise")
        noise = max(noise_floor, float(_rep)) if _rep else noise_floor
        verdict = decide_verdict(correctness, delta, noise, pre_green=not green,
                                 claim=str(ev.get("claim") or ""), changed_files=files)
        if verdict == "INVALID_NOOP":
            backend.reset_to(head)
            noops += 1
        elif correctness is False:
            backend.reset_to(head)
            impl_fails += 1
        else:
            head = commit
            impl_fails = 0

        plateau = plateau + 1 if verdict != "KEEP" else 0
        rec = IterationRecord(
            iter=i, round_id="r%d" % i, change=change or "",
            changed_files=list(files),
            workspace_commit=commit, correctness=correctness, verdict=verdict,
            base_ms=base_ms, cand_ms=cand_ms,
            vs_base=(1.0 + delta) if delta is not None else None,
            utilization=ev.get("utilization"), floor_us=ev.get("floor_us"),
            acc=dict(ev.get("acc") or {}), claim=str(ev.get("claim") or ""),
            refuted_directions=list(ev.get("refuted_directions") or []),
            noise_floor=noise,
            agent_meta=dict(getattr(backend, "last_agent_meta", {}) or {}),
        )
        records.append(rec)
        if on_round is not None:
            # 每轮立即落盘：崩溃/超时最多丢当前这一轮，不会丢整次运行
            on_round(rec)
        if delta is not None:
            ratios.append(1.0 + delta)

        if noops >= MAX_NOOP:
            rec.stop_reason = ("agent_timeout"
                               if str(change).startswith("[agent-timeout") else "noop")
            break
        if impl_fails >= max_impl_fails:
            rec.stop_reason = "impl_fails"
            break
        if plateau >= max_plateau:
            rec.stop_reason = "plateau"
            break
        if i == rounds:
            rec.stop_reason = "rounds_done"
    return records


def _selfcheck() -> int:
    problems = []
    fields = set(IterationRecord.__dataclass_fields__)
    need = {"iter", "round_id", "change", "changed_files", "workspace_commit",
            "correctness", "verdict", "base_ms", "cand_ms", "vs_base",
            "utilization", "floor_us", "acc", "claim", "refuted_directions",
            "noise_floor"}
    missing = need - fields
    if missing:
        problems.append("IterationRecord 缺字段: %s" % sorted(missing))
    for m in ("run_agent", "commit", "reset_to", "eval"):
        if not hasattr(Backend, m):
            problems.append("Backend 缺方法 %s" % m)
    if problems:
        print("✗ selfcheck: %s" % "; ".join(problems))
        return 1
    print("✓ selfcheck: IterationRecord(%d 字段) + Backend(4 方法) 自洽" % len(fields))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="driver", description="P3 driver（控制流 + backend）")
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--demo", type=int, default=0)
    p.add_argument("--fake-flat", action="store_true")
    p.add_argument("--fake-fail-every", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--render-demo", action="store_true")
    p.add_argument("--init-workspace", default=None)
    p.add_argument("--real", action="store_true")
    p.add_argument("--workspace", default=None)
    p.add_argument("--card-rel", default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--agent-cmd", default=None)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--agent-timeout", type=int, default=1200,
                   help="单轮 agent 的 wall-clock 上限（秒）；默认 20 分钟")
    a = p.parse_args(argv)

    if a.selfcheck:
        return _selfcheck()
    if a.render_demo:
        print(render_student_prompt([], "demo"))
        return 0
    if a.init_workspace:
        import shutil
        dst = Path(a.init_workspace).resolve()
        if dst.exists():
            print("✗ 目标已存在：%s" % dst)
            return 2
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(Path(__file__).resolve().parents[3], dst,
                        ignore=shutil.ignore_patterns(".git", ".artifacts", "__pycache__"))
        print("✓ 工作区：%s" % dst)
        return 0
    if a.real:
        if not (a.workspace and a.card_rel and a.tag and a.agent_cmd):
            print("✗ --real 需要 --workspace --card-rel --tag --agent-cmd", file=sys.stderr)
            return 2
        be = RealBackend(Path(a.workspace), a.card_rel, a.tag, a.agent_cmd,
                         gpu=a.gpu, timeout_s=a.agent_timeout, run_id="p32")
        # 预检：--tag 必须能在 candidates.yaml 里找到候选，否则整轮 agent 会
        # 收到一个"没有目标"的任务（实测：它会花满 20 分钟读仓库，0 次 Write）。
        cand = None
        try:
            import yaml
            cd = yaml.safe_load((be.card / "candidates.yaml").read_text(
                encoding="utf-8")) or {}
            cands = [x for x in (cd.get("candidates") or []) if isinstance(x, dict)]
            cand = next((x for x in cands if str(x.get("id")) == str(a.tag)), None)
            if cand is None:
                print("✗ --tag %r 不在 %s/candidates.yaml 里（现有：%s）。"
                      "先补候选，别让 agent 空跑。"
                      % (a.tag, be.card, ", ".join(str(x.get("id")) for x in cands)),
                      file=sys.stderr)
                return 2
            if not cand.get("variant"):
                print("✗ 候选 %r 没有声明 variant —— loop.py run 会拒绝" % a.tag,
                      file=sys.stderr)
                return 2
        except FileNotFoundError:
            print("✗ 缺少 %s/candidates.yaml" % be.card, file=sys.stderr)
            return 2
        be.init_workspace()
        # 先把落盘路径定下来：每轮结束立刻追加（不再攒到最后）
        out = Path(a.out) if a.out else (
            repo_records_dir(be.ws) / ("iterate-%s.jsonl" % time.strftime("%Y%m%d_%H%M%S")))
        out.parent.mkdir(parents=True, exist_ok=True)

        def _append(rec):
            with out.open("a", encoding="utf-8") as f:
                f.write(rec.to_json() + "\n")

        recs = iterate(be, a.demo or 2, card=a.card_rel, candidate=cand,
                       on_round=_append)
        for r in recs:
            print("r%-2d %-12s cand=%s base=%s claim=%s%s"
                  % (r.iter, r.verdict, r.cand_ms, r.base_ms, r.claim,
                     ("  stop=" + r.stop_reason) if r.stop_reason else ""))
        print("→ %s（%d 条，逐轮追加）" % (out, len(recs)))
        return 0
    if a.demo:
        be = FakeBackend(flat=a.fake_flat, fail_every=a.fake_fail_every)
        recs = iterate(be, a.demo, card="demo")
        for r in recs:
            print("r%-2d %-7s cand=%.3f base=%.3f claim=%s%s"
                  % (r.iter, r.verdict, r.cand_ms or 0, r.base_ms or 0, r.claim,
                     ("  stop=" + r.stop_reason) if r.stop_reason else ""))
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text("\n".join(r.to_json() for r in recs) + "\n",
                                   encoding="utf-8")
            print("→ %s（%d 条）" % (a.out, len(recs)))
        return 0
    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
