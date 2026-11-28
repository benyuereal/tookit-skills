#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3 driver：一轮的定义 + 可注入 backend（P3.0 契约 / P3.1 控制流）。

设计见 `updates/handoff/HANDOFF-2026-11-28-P3-driver-plan.md` 与
`updates/handoff/P3-CONTRACT.md`。

**本文件只做两件事**：
  1. 定义一轮的产物 `IterationRecord` 和副作用接缝 `Backend`（P3.0）；
  2. 用 `FakeBackend` 把控制流跑通并单测（P3.1）——**不碰 GPU、不调 claude**。

真正的 backend（Claude 子进程 / GPU / git）是 P3.2，不在本文件里。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Protocol

AGENT_PATHS = ("Plugins/ops/skills/",)   # 只认 agent 该改的地方
ENGINE_PATHS = ("memory/", "results/", ".artifacts/", "Plugins/ops/op-atlas/")
MAX_NOOP = 1

NOISE_FLOOR = 0.015          # 1.5%：低于这个的增益不算提升
MAX_PLATEAU = 3              # 连续这么多轮没有 KEEP → plateau
MAX_IMPL_FAILS = 3           # 连续这么多轮正确性破坏 → 停


# ---------------------------------------------------------------- P3.0 契约

@dataclass
class IterationRecord:
    """一轮的产物。字段先少后加（P3.4 再补渲染需要的）。"""

    iter: int
    round_id: str
    change: str                       # agent 自述「改了什么」，不含数字
    changed_files: list[str] = field(default_factory=list)
    workspace_commit: str = ""        # 有了它才能 revert
    correctness: bool | None = None   # None = pre-green，尚未评估
    verdict: str = "NEUTRAL"          # KEEP | REVERT | NEUTRAL | REFUTED
    base_ms: float | None = None
    cand_ms: float | None = None
    vs_base: float | None = None      # 同 job back-to-back 比值
    utilization: float | None = None
    floor_us: float | None = None
    acc: dict = field(default_factory=dict)
    claim: str = ""                   # PASS | FAIL:<reason>
    refuted_directions: list[str] = field(default_factory=list)
    noise_floor: float = NOISE_FLOOR
    stop_reason: str | None = None    # plateau | impl_fails | rounds_done

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class Backend(Protocol):
    """副作用接缝（G2 + G9）。控制流只依赖这四个方法，所以可以 fake。"""

    def run_agent(self, prompt: str, role: str = "student") -> str:
        """调 agent；返回它自述的「改了什么」。**不得从返回值解析数字**。"""
        ...

    def commit(self, message: str) -> str:
        """提交当前工作区；返回 commit hash。"""
        ...

    def reset_to(self, commit: str) -> None:
        """把工作区回滚到指定 commit（只在正确性破坏时调用）。"""
        ...

    def eval(self, tag: str) -> dict:
        """跑测量，返回
        {"cand_ms", "base_ms", "correctness": bool, "acc": {...},
         "claim": "PASS"|"FAIL:...", "utilization", "floor_us",
         "refuted_directions": [...]}
        """
        ...


# ---------------------------------------------------------------- 渲染

def render_student_prompt(records: list[IterationRecord], card: str = "",
                          candidate: dict | None = None) -> str:
    """下一轮 prompt：上一轮结构化记录 + 死路清单 + 自己改过什么。"""
    lines = [
        "# 算子实现任务%s" % (("：" + card) if card else ""),
        "",
        "你只负责**写实现**。测量、判定、记账全部由引擎完成；"
        "不要在会话里自己跑 bench 或计时。",
        "",
        "## 上一轮（结构化，不是总结）",
    ]
    if not records:
        lines.append("（第一轮，无历史）")
    else:
        r = records[-1]
        lines.append(json.dumps({
            "round": r.iter,
            "verdict": r.verdict,
            "change": r.change,
            "changed_files": r.changed_files,
            "vs_base": r.vs_base,
            "cand_ms": r.cand_ms,
            "base_ms": r.base_ms,
            "utilization": r.utilization,
            "claim": r.claim,
            "acc": r.acc,
        }, ensure_ascii=False, indent=2))
    refuted: list[str] = []
    for r in records:
        refuted.extend(r.refuted_directions or [])
    if candidate:
        lines += ["", "## 本轮要实现的候选",
                  json.dumps({k: candidate.get(k) for k in
                              ("id", "variant", "direction", "expect",
                               "hypothesis", "expect_basis") if candidate.get(k) is not None},
                             ensure_ascii=False, indent=2)]
    lines += ["", "## 已证伪方向（refuted_directions，不要重复试）",
              json.dumps(sorted(set(refuted)), ensure_ascii=False)]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- P3.1 控制流

def agent_changes(changed_files: list[str]) -> list[str]:
    """只保留 agent 该改的路径。引擎自己写的一律不算。"""
    return [f for f in (changed_files or [])
            if any(f.startswith(p) for p in AGENT_PATHS)
            and not any(f.startswith(q) for q in ENGINE_PATHS)]


def _verdict(correctness: bool | None, delta: float | None, noise: float,
             pre_green: bool, claim: str = "", changed_files=None) -> str:
    """relaxed-revert：**性能回退保留**，只有正确性破坏才 REVERT。"""
    # **noop 守卫**：agent 什么都没改，绝不允许产生 KEEP（更不允许晋升）。
    # 实测 2026-11-28：agent=noop、changed_files=[] 却判 KEEP，因为它把卡里
    # 早有的实现与一个慢基线比了一次 —— KEEP 就失去了含义。
    if not agent_changes(list(changed_files or [])):
        return "INVALID_NOOP"
    if correctness is False:
        return "REVERT"
    # claim 的失败项是精度/证据 → 不是"白跑一趟"，是"这个实现不能声称正确"
    if claim.startswith(("FAIL:accuracy", "FAIL:evidence")):
        return "REVERT"
    if pre_green or correctness is None:
        return "NEUTRAL"          # pre-green 语义：还没通过正确性，不评估性能
    if delta is not None and delta < -noise:
        return "KEEP"
    return "NEUTRAL"


def iterate(backend: Backend, rounds: int, card: str = "",
            noise_floor: float = NOISE_FLOOR, max_plateau: int = MAX_PLATEAU,
            max_impl_fails: int = MAX_IMPL_FAILS,
            candidate: dict | None = None) -> list[IterationRecord]:
    """跑 N 轮。**纯控制流**：所有副作用都经 backend，因此可用 fake 单测。"""
    records: list[IterationRecord] = []
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
        if hasattr(backend, "no_promote"):
            backend.no_promote = bool(noop)    # noop 轮禁止晋升
        ev = backend.eval(("r%d" % i))

        correctness = ev.get("correctness")
        if correctness is True:
            green = True
        base_ms = ev.get("base_ms")
        cand_ms = ev.get("cand_ms")
        delta = None
        if base_ms and cand_ms:
            delta = cand_ms / base_ms - 1.0

        verdict = _verdict(correctness, delta, noise_floor, pre_green=not green,
                           claim=str(ev.get("claim") or ""), changed_files=files)
        if verdict == "INVALID_NOOP":
            backend.reset_to(head)          # 没有 agent 改动：丢弃空提交
            noops += 1
        elif correctness is False:
            backend.reset_to(head)          # 只有正确性破坏才回滚
            impl_fails += 1
        else:
            head = commit
            impl_fails = 0

        plateau = plateau + 1 if verdict != "KEEP" else 0
        rec = IterationRecord(
            iter=i,
            round_id="r%d" % i,
            change=change or "",
            changed_files=list(ev.get("changed_files") or []),
            workspace_commit=commit,
            correctness=correctness,
            verdict=verdict,
            base_ms=base_ms,
            cand_ms=cand_ms,
            vs_base=(1.0 + delta) if delta is not None else None,
            utilization=ev.get("utilization"),
            floor_us=ev.get("floor_us"),
            acc=dict(ev.get("acc") or {}),
            claim=str(ev.get("claim") or ""),
            refuted_directions=list(ev.get("refuted_directions") or []),
            noise_floor=noise_floor,
        )
        records.append(rec)

        if noops >= MAX_NOOP:
            rec.stop_reason = "noop"
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


# ---------------------------------------------------------------- fake backend

class FakeBackend:
    """确定性 fake：不跑 GPU、不调 agent、不做 git。

    `flat=True` 时每轮都只有 0.5% 的改善（低于噪声底），用来验收 plateau。
    `evidence="self-consistent-only"` 时 claim FAIL，用来验收 REVERT 门。
    """

    def __init__(self, flat: bool = False, evidence: str = "vendor_library",
                 fail_every: int = 0, times=None, green_from: int = 1,
                 noop: bool = False):
        self.flat = flat
        self.evidence = evidence
        self.fail_every = fail_every
        self.green_from = green_from
        self.noop = noop
        self._changed = [] if noop else ["Plugins/ops/skills/fake/_common.py"]
        self.n = 0
        self.commits: list[str] = ["c0"]
        self.reset_calls: list[str] = []
        self.agent_calls: list[str] = []
        self.times = list(times or [])

    def run_agent(self, prompt: str, role: str = "student") -> str:
        self.agent_calls.append(prompt)
        return "candidate change #%d" % (len(self.agent_calls))

    def commit(self, message: str) -> str:
        sha = "c%d" % len(self.commits)
        self.commits.append(sha)
        return sha

    @property
    def changed_files(self) -> list[str]:
        return list(self._changed)

    def reset_to(self, commit: str) -> None:
        self.reset_calls.append(commit)

    def eval(self, tag: str) -> dict:
        self.n += 1
        i = self.n
        base = 0.200
        if self.times:
            cand = self.times[min(i - 1, len(self.times) - 1)]
        elif self.flat:
            cand = base * 0.995
        else:
            cand = base * (0.80 ** i)      # 每轮 -20%
        if i < self.green_from:
            correctness = None          # pre-green：还没通过正确性
        else:
            correctness = not (self.fail_every and i % self.fail_every == 0)
        claim = "PASS" if self.evidence in ("vendor_library", "independent_reference") \
            else "FAIL:evidence_unverified"
        return {
            "cand_ms": cand,
            "base_ms": base,
            "correctness": correctness,
            "acc": {"max_ulp": 0.5, "cosine": 0.999998,
                    "evidence_kind": self.evidence},
            "claim": claim,
            "utilization": 0.82,
            "floor_us": 116.4,
            "refuted_directions": ["fusion-boundary"] if i >= 2 else [],
        }


# ---------------------------------------------------------------- P3.2 真 backend

class RealBackend:
    """真 backend：git 工作区 + agent 子进程 + 引擎测量（loop.py run/claim）。

    工作区是**整仓副本**（`.artifacts/<arm>/<op>_<id>/workspace/`），这样
    `card.parents[3]/scripts` 等相对路径仍然成立，且 git 历史只属于这一次 run。
    """

    def __init__(self, workspace: Path, card_rel: str, tag: str, agent_cmd: str,
                 gpu: int = 0, timeout_s: int = 1800, run_id: str = "p32"):
        self.ws = Path(workspace).resolve()
        self.card_rel = card_rel
        self.tag = tag
        self.agent_cmd = agent_cmd
        self.gpu = str(gpu)
        self.timeout_s = timeout_s
        self.run_id = run_id
        self.card = self.ws / card_rel
        self.artifacts = self.ws / ".artifacts"
        self.env = dict(os.environ,
                        OPS_GPU=self.gpu, HIP_VISIBLE_DEVICES=self.gpu,
                        CUDA_VISIBLE_DEVICES=self.gpu, OPS_ARM="driver",
                        OPS_RUN_ID=run_id,
                        OPS_ARTIFACTS=str(self.artifacts))
        self._claim = ""
        self._changed: list[str] = []
        self.no_promote = False

    @property
    def changed_files(self) -> list[str]:
        return list(self._changed)

    # -- git
    def _git(self, *args, check=True):
        import subprocess
        # safe.directory：工作区可能被 chown 给 agent 用户，root 跑 git 会因
        # "dubious ownership" 拒绝（实测 2026-11-28）。
        return subprocess.run(["git", "-c", "safe.directory=*",
                               "-C", str(self.ws), *args],
                              capture_output=True, text=True, check=check)

    def init_workspace(self):
        """首次使用：git init + 初始提交（工作区必须已经是仓库副本）。"""
        if not (self.ws / ".git").is_dir():
            self._git("init", "-q")
            self._git("config", "user.email", "driver@local")
            self._git("config", "user.name", "driver")
        self.commit("initial")

    def commit(self, message: str) -> str:
        self._git("add", "-A")
        # 允许空提交：agent 说"没改"也要有一个可回滚的点
        self._git("-c", "user.email=driver@local", "-c", "user.name=driver",
                  "commit", "-q", "--allow-empty", "-m", message or "round")
        out = self._git("rev-parse", "HEAD").stdout.strip()
        try:
            self._changed = [x for x in self._git(
                "show", "--name-only", "--pretty=format:", "HEAD").stdout.splitlines()
                if x and not any(x.startswith(q) for q in ENGINE_PATHS)]
        except Exception:                                    # noqa: BLE001
            self._changed = []
        return out

    def reset_to(self, commit: str) -> None:
        self._git("reset", "--hard", commit)

    # -- agent
    def run_agent(self, prompt: str, role: str = "student") -> str:
        import subprocess
        r = subprocess.run(self.agent_cmd, shell=True, cwd=str(self.ws),
                           input=prompt, text=True, timeout=self.timeout_s,
                           env=self.env, capture_output=True)
        if r.returncode != 0:
            raise RuntimeError("agent rc=%d: %s" % (r.returncode, r.stderr[-500:]))
        return r.stdout.strip()

    # -- eval
    def _loop(self, *args):
        import subprocess
        env = dict(self.env)
        if self.no_promote:
            env["OPS_NO_PROMOTE"] = "1"
        return subprocess.run(
            [sys.executable, "-I", str(self.ws / "Plugins/ops/engine/loop.py"),
             "-C", str(self.card), *args],
            capture_output=True, text=True, env=env, timeout=3600)

    def _ensure_judge(self) -> None:
        """跑 run 之前确保 judge/冻结容差存在（幂等）。

        没有它 → loop.py 退回 contract 容差 → claim 第五项 FAIL:evidence →
        每轮必然 REVERT，闭环永远不"接受"任何结果（lead 2026-11-28 打回）。
        judge 产物放本次 run 的 artifact 目录，避免写脏卡片。
        """
        import subprocess
        jd = (self.artifacts / "driver"
              / ("%s_%s" % (self.card.name, self.run_id)) / "judge")
        if (jd / "tolerance.json").is_file():
            return
        subprocess.run(
            [sys.executable, "-I", str(self.ws / "Plugins/ops/engine/judge.py"),
             "generate", "--card", str(self.card), "--out", str(jd)],
            cwd=str(self.ws), env=self.env, capture_output=True)

    def eval(self, tag: str) -> dict:
        import glob
        self._ensure_judge()
        bench = self.ws / "results" / "bench_std.json"
        before = set(glob.glob(str(self.artifacts / "**/runs/loop/*.jsonl"), recursive=True))
        r = self._loop("run", self.tag, "--bench-std", str(bench))
        self._claim = ""
        if r.returncode == 2:
            return {"correctness": False, "claim": "FAIL:run_rejected",
                    "cand_ms": None, "base_ms": None,
                    "changed_files": self._changed,
                    "refuted_directions": []}
        # 只读**本次 run** 的账本：全局 glob 会把上一次 run（甚至别的卡）的记录
        # 当成这一轮的结果（实测 2026-11-28：silu 与 newop 读出了同一组数字）。
        run_dir = (self.artifacts / "driver"
                   / ("%s_%s" % (self.card.name, self.run_id)) / "runs" / "loop")
        recs = []
        for f in glob.glob(str(run_dir / "*.jsonl")):
            try:
                for line in open(f, encoding="utf-8", errors="replace"):
                    line = line.strip()
                    if line.startswith("{"):
                        recs.append(json.loads(line))
            except (OSError, ValueError):
                continue
        rec = recs[-1] if recs else {}
        # 记录里的 tag 是**变体名**，CLI 的 --tag 是候选 id；先按候选 id，
        # 找不到再用变体名重试（与 iterate.py 同口径）。
        c = self._loop("claim", "--tag", self.tag)
        if "没有 tag=" in c.stdout:
            try:
                import yaml
                cd = yaml.safe_load((self.card / "candidates.yaml").read_text(
                    encoding="utf-8")) or {}
                for x in cd.get("candidates") or []:
                    if str(x.get("id")) == str(self.tag) and x.get("variant"):
                        c = self._loop("claim", "--tag", str(x["variant"]))
                        break
            except Exception:                                # noqa: BLE001
                pass
        if "结论：**PASS**" in c.stdout:
            claim = "PASS"
        else:
            # 把失败项分类，verdict 才能按计划区分 REVERT 的原因
            fails = [ln for ln in c.stdout.splitlines() if ln.strip().startswith("✗")]
            evk = str(rec.get("evidence_kind")
                      or (rec.get("acc") or {}).get("evidence_kind") or "")
            if (any("精度" in ln or "证据" in ln for ln in fails)
                    or evk not in ("independent_reference", "vendor_library")):
                claim = "FAIL:evidence"
            elif any("口径" in ln or "显著性" in ln for ln in fails):
                claim = "FAIL:caliber"
            else:
                claim = "FAIL:claim"
        return {
            "cand_ms": rec.get("cand_ms"),
            "base_ms": rec.get("base_ms"),
            "correctness": (rec.get("decision") or "").startswith("ACCEPTED")
                            or rec.get("decision") in ("INCONCLUSIVE", "CALIBRATION_ONLY"),
            "acc": rec.get("acc") or {},
            "claim": claim,
            "utilization": None,
            "floor_us": None,
            "refuted_directions": [],
            "changed_files": self._changed,
            "decision": rec.get("decision"),
        }


# ---------------------------------------------------------------- CLI

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
    print("✓ selfcheck: IterationRecord(%d 字段) + Backend(4 方法) 自洽"
          % len(fields))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="driver", description="P3 driver（控制流 + fake）")
    p.add_argument("--selfcheck", action="store_true")
    p.add_argument("--demo", type=int, default=0, help="用 FakeBackend 跑 N 轮")
    p.add_argument("--fake-flat", action="store_true", help="每轮只有噪声内改善")
    p.add_argument("--fake-fail-every", type=int, default=0,
                   help="每 N 轮制造一次正确性破坏")
    p.add_argument("--out", default=None, help="把 IterationRecord 写成 jsonl")
    p.add_argument("--render-demo", action="store_true", help="打印一次学生 prompt")
    p.add_argument("--init-workspace", default=None,
                   help="把仓库复制到该目录并 git init（P3.2 真 backend 用）")
    p.add_argument("--real", action="store_true", help="真 backend（git + agent + loop.py）")
    p.add_argument("--workspace", default=None)
    p.add_argument("--card-rel", default=None, help="工作区内的卡相对路径")
    p.add_argument("--tag", default=None, help="要跑的候选 id")
    p.add_argument("--agent-cmd", default=None)
    p.add_argument("--gpu", type=int, default=0)
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
            print("✗ 目标已存在：%s" % dst); return 2
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(ROOT if "ROOT" in globals() else Path(__file__).resolve().parents[3],
                        dst, ignore=shutil.ignore_patterns(".git", ".artifacts", "__pycache__"))
        print("✓ 工作区：%s" % dst)
        return 0
    if a.real:
        if not (a.workspace and a.card_rel and a.tag and a.agent_cmd):
            print("✗ --real 需要 --workspace --card-rel --tag --agent-cmd", file=sys.stderr)
            return 2
        be = RealBackend(Path(a.workspace), a.card_rel, a.tag, a.agent_cmd,
                         gpu=a.gpu, run_id="p32")
        be.init_workspace()
        cand = None
        try:
            import yaml
            cd = yaml.safe_load((be.card / "candidates.yaml").read_text(
                encoding="utf-8")) or {}
            cand = next((x for x in (cd.get("candidates") or [])
                         if str(x.get("id")) == str(a.tag)), None)
        except Exception:                                    # noqa: BLE001
            cand = None
        recs = iterate(be, a.demo or 2, card=a.card_rel, candidate=cand)
        for r in recs:
            print("r%-2d %-7s cand=%s base=%s claim=%s%s"
                  % (r.iter, r.verdict, r.cand_ms, r.base_ms, r.claim,
                     ("  stop=" + r.stop_reason) if r.stop_reason else ""))
        if a.out:
            Path(a.out).write_text("\n".join(r.to_json() for r in recs) + "\n",
                                   encoding="utf-8")
            print("→ %s（%d 条）" % (a.out, len(recs)))
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
            Path(a.out).write_text(
                "\n".join(r.to_json() for r in recs) + "\n", encoding="utf-8")
            print("→ %s（%d 条）" % (a.out, len(recs)))
        return 0
    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
