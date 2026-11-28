#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""三类 backend：fake（单测）、真工作区（git + agent + 引擎测量）。

**一个类只干一件事**（micro-vllm 风格）：
    WorkspaceGit    只负责工作区的 git 语义（commit / reset / changed_files）
    ClaudeAgent     只负责调 agent 子进程，并把 `claude -p --output-format json`
                    的 stdout 解析成 result + 诊断字段
    EngineEvaluator 只负责调 loop.py（run/claim）并读回本次 run 的记录
    RealBackend     门面：把上面三个组合起来，实现 contract.Backend

FakeBackend 留在本文件最前面：它与真 backend 共用同一组契约，便于对照。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from contract import (Backend, ENGINE_PATHS, NOISE_FLOOR,  # noqa: F401
                      correctness_of_decision)


class FakeBackend:
    """确定性 fake：不跑 GPU、不调 agent、不做 git。"""

    def __init__(self, flat: bool = False, evidence: str = "vendor_library",
                 fail_every: int = 0, times=None, green_from: int = 1,
                 noop: bool = False, noise: float | None = None,
                 tamper: bool = False, work: bool = True):
        self.flat = flat
        self.evidence = evidence
        self.fail_every = fail_every
        self.green_from = green_from
        self.noop = noop
        self.noise = noise
        self.tamper = tamper
        self.work = work
        if tamper:
            self._changed = ["Plugins/ops/engine/loop.py"]
        else:
            self._changed = [] if noop else ["Plugins/ops/skills/fake/_common.py"]
        self.n = 0
        self.commits: list[str] = ["c0"]
        self.reset_calls: list[str] = []
        self.agent_calls: list[str] = []
        self.times = list(times or [])
        self.last_agent_meta: dict = {}

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

    def next_candidate(self):
        """P0-2：fake 里模拟"队列还有/没有未实现候选"。"""
        return {"id": "fake", "variant": "fake-v"} if self.work else None

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
            cand = base * (0.80 ** i)
        if i < self.green_from:
            correctness = None
        else:
            correctness = not (self.fail_every and i % self.fail_every == 0)
        claim = "PASS" if self.evidence in ("vendor_library", "independent_reference") \
            else "FAIL:evidence_unverified"
        return {
            "cand_ms": cand, "base_ms": base, "correctness": correctness,
            "acc": {"max_ulp": 0.5, "cosine": 0.999998,
                    "evidence_kind": self.evidence},
            "claim": claim, "utilization": 0.82, "floor_us": 116.4,
            "refuted_directions": ["fusion-boundary"] if i >= 2 else [],
            "noise": self.noise,
        }


class WorkspaceGit:
    """工作区的 git 语义。只做 commit/reset/diff，不关心 agent 与测量。"""

    def __init__(self, ws: Path):
        self.ws = Path(ws).resolve()
        self._changed: list[str] = []

    def _git(self, *args, check=True):
        # safe.directory：工作区可能被 chown 给 agent 用户，root 跑 git 会因
        # "dubious ownership" 拒绝（实测 2026-11-28）。
        return subprocess.run(["git", "-c", "safe.directory=*",
                               "-C", str(self.ws), *args],
                              capture_output=True, text=True, check=check)

    def init(self):
        if not (self.ws / ".git").is_dir():
            self._git("init", "-q")
            self._git("config", "user.email", "driver@local")
            self._git("config", "user.name", "driver")
        self.commit("initial")

    def commit(self, message: str) -> str:
        self._git("add", "-A")
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

    @property
    def changed_files(self) -> list[str]:
        return list(self._changed)


class ClaudeAgent:
    """调 agent 子进程。stdout 若是 claude 的 JSON，取 result 并留诊断字段。"""

    def __init__(self, cmd: str, cwd: Path, env: dict, timeout_s: int = 1200):
        self.cmd = cmd
        self.cwd = Path(cwd)
        self.env = env
        self.timeout_s = timeout_s
        self.last_agent_meta: dict = {}

    def run(self, prompt: str) -> str:
        try:
            r = subprocess.run(self.cmd, shell=True, cwd=str(self.cwd),
                               input=prompt, text=True, timeout=self.timeout_s,
                               env=self.env, capture_output=True)
        except subprocess.TimeoutExpired as e:
            part = ""
            try:
                part = (e.stdout or b"").decode("utf-8", "replace")[-400:]
            except Exception:                                # noqa: BLE001
                part = ""
            return "[agent-timeout after %ss] partial=%s" % (self.timeout_s, part)
        if r.returncode != 0:
            return "[agent-rc=%d] %s" % (r.returncode, (r.stderr or "")[-300:])
        out = (r.stdout or "").strip()
        self.last_agent_meta = {}
        try:
            obj = json.loads(out)
            if isinstance(obj, dict):
                self.last_agent_meta = {k: obj.get(k) for k in
                                        ("duration_api_ms", "num_turns",
                                         "total_cost_usd", "is_error", "session_id")
                                        if obj.get(k) is not None}
                res = obj.get("result")
                if isinstance(res, str) and res.strip():
                    return res.strip()[:4000]
        except (ValueError, TypeError):
            pass
        return out[:2000]


class EngineEvaluator:
    """调 loop.py run/claim，读回本次 run 的记录。"""

    def __init__(self, ws: Path, card: Path, tag: str, run_id: str,
                 env: dict, no_promote: bool = False):
        self.ws = Path(ws).resolve()
        self.card = Path(card)
        self.tag = tag
        self.run_id = run_id
        self.env = env
        self.no_promote = no_promote

    def _loop(self, *args):
        env = dict(self.env)
        if self.no_promote:
            env["OPS_NO_PROMOTE"] = "1"
        return subprocess.run(
            [sys.executable, "-I", str(self.ws / "Plugins/ops/engine/loop.py"),
             "-C", str(self.card), *args],
            capture_output=True, text=True, env=env, timeout=3600)

    def _artifact_root(self) -> Path:
        return (self.ws / ".artifacts" / "driver"
                / ("%s_%s" % (self.card.name, self.run_id)))

    def ensure_judge(self) -> None:
        jd = self._artifact_root() / "judge"
        if (jd / "tolerance.json").is_file():
            return
        subprocess.run(
            [sys.executable, "-I", str(self.ws / "Plugins/ops/engine/judge.py"),
             "generate", "--card", str(self.card), "--out", str(jd)],
            cwd=str(self.ws), env=self.env, capture_output=True)

    def _read_repeat_record(self) -> dict:
        run_dir = self._artifact_root() / "runs" / "loop"
        recs = []
        for f in sorted(run_dir.glob("*.jsonl")):
            try:
                for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
                    line = line.strip()
                    if line.startswith("{"):
                        recs.append(json.loads(line))
            except (OSError, ValueError):
                continue
        return recs[-1] if recs else {}

    def _claim_for(self, variant_hint: str, tag: str | None = None) -> str:
        c = self._loop("claim", "--tag", str(tag or self.tag))
        if "没有 tag=" in c.stdout and variant_hint:
            c = self._loop("claim", "--tag", variant_hint)
        if "结论：**PASS**" in c.stdout:
            return "PASS"
        fails = [ln for ln in c.stdout.splitlines() if ln.strip().startswith("✗")]
        evk = str(self._rec.get("evidence_kind")
                  or (self._rec.get("acc") or {}).get("evidence_kind") or "")
        if (any("精度" in ln or "证据" in ln for ln in fails)
                or evk not in ("independent_reference", "vendor_library")):
            return "FAIL:evidence"
        if any("口径" in ln or "显著性" in ln for ln in fails):
            return "FAIL:caliber"
        return "FAIL:claim"

    def evaluate(self, tag: str | None = None) -> dict:
        tag = str(tag or self.tag)
        self.ensure_judge()
        bench = self.ws / "results" / "bench_std.json"
        rep = os.environ.get("OPS_REPEAT", "3")
        r = self._loop("run", tag, "--bench-std", str(bench), "--repeat", str(rep))
        if r.returncode == 2:
            return {"correctness": False, "claim": "FAIL:run_rejected",
                    "cand_ms": None, "base_ms": None,
                    "refuted_directions": []}
        self._rec = self._read_repeat_record()
        variant = ""
        try:
            import yaml
            cd = yaml.safe_load((self.card / "candidates.yaml").read_text(
                encoding="utf-8")) or {}
            for x in cd.get("candidates") or []:
                if str(x.get("id")) == str(tag) and x.get("variant"):
                    variant = str(x["variant"])
                    break
        except Exception:                                    # noqa: BLE001
            pass
        claim = self._claim_for(variant, tag)
        rec = self._rec
        return {
            "cand_ms": rec.get("cand_ms"), "base_ms": rec.get("base_ms"),
            "correctness": correctness_of_decision(rec.get("decision")),
            "acc": rec.get("acc") or {}, "claim": claim,
            "utilization": None, "floor_us": None, "refuted_directions": [],
            "decision": rec.get("decision"), "noise": rec.get("repeat_spread"),
        }


class RealBackend:
    """门面：组合 git / agent / evaluator，实现 contract.Backend。"""

    def __init__(self, workspace: Path, card_rel: str, tag: str, agent_cmd: str,
                 gpu: int = 0, timeout_s: int = 1200, run_id: str = "p32",
                 system_prompt_file: str | None = None):
        self.ws = Path(workspace).resolve()
        self.card = self.ws / card_rel
        self.run_id = run_id
        (self.ws / ".scratch").mkdir(exist_ok=True)
        # 全局「实现者契约」：由库存 wrapper 通过 --append-system-prompt 注入
        _sys = (system_prompt_file
                or os.environ.get("OPS_SYSTEM_PROMPT_FILE")
                or str(Path(__file__).resolve().parent
                       / "instrument" / "prompts" / "system_student.md"))
        self.env = dict(os.environ,
                        OPS_GPU=str(gpu), HIP_VISIBLE_DEVICES=str(gpu),
                        CUDA_VISIBLE_DEVICES=str(gpu), OPS_ARM="driver",
                        OPS_RUN_ID=run_id, OPS_WS=str(self.ws),
                        OPS_ENV_FILE=os.environ.get("OPS_ENV_FILE",
                                                    "/workspace/tsoc.env"),
                        OPS_SYSTEM_PROMPT_FILE=_sys,
                        OPS_ARTIFACTS=str(self.ws / ".artifacts"))
        self.git = WorkspaceGit(self.ws)
        self.agent = ClaudeAgent(agent_cmd, self.ws, self.env, timeout_s)
        self.evaluator = EngineEvaluator(self.ws, self.card, tag, run_id, self.env)
        self.no_promote = False

    # -- 门面转发
    def init_workspace(self):
        self.git.init()

    def commit(self, message: str) -> str:
        return self.git.commit(message)

    def reset_to(self, commit: str) -> None:
        self.git.reset_to(commit)

    def run_agent(self, prompt: str, role: str = "student") -> str:
        return self.agent.run(prompt)

    def eval(self, tag: str) -> dict:
        self.evaluator.no_promote = self.no_promote
        return self.evaluator.evaluate(tag)

    def _implemented_variants(self) -> set:
        """_common.py 里已登记（VARIANTS / HIP_VARIANTS）或 forged 的变体名。"""
        import re as _re
        got = set()
        common = self.card / "_common.py"
        if common.is_file():
            src = common.read_text(encoding="utf-8", errors="replace")
            m = _re.search(r"^VARIANTS\s*=\s*\(([^)]*)\)", src, _re.M)
            if m:
                got |= {x.strip().strip("'\"") for x in m.group(1).split(",") if x.strip()}
            m2 = _re.search(r"^HIP_VARIANTS\s*=\s*\{(.*?)\}", src, _re.M | _re.S)
            if m2:
                got |= set(_re.findall(r'"([^"]+)"\s*:', m2.group(1)))
        forged = self.card / "runs" / "forged"
        if forged.is_dir():
            got |= {p.stem for p in forged.glob("*.py")}
        return got

    def next_candidate(self):
        """下一个**未实现**的 pending 候选；没有就返回 None（→ NO_WORK）。

        这样队列耗尽时不再把 agent 叫起来空转 20 分钟（lead 2026-11-28）。
        """
        try:
            import yaml
            cd = yaml.safe_load((self.card / "candidates.yaml").read_text(
                encoding="utf-8")) or {}
        except Exception:                                    # noqa: BLE001
            return None
        impl = self._implemented_variants()
        for c in cd.get("candidates") or []:
            if not isinstance(c, dict):
                continue
            if str(c.get("status") or "") in ("done", "implemented"):
                continue
            v = str(c.get("variant") or "")
            if v and v not in impl:
                return c
        return None

    @property
    def changed_files(self) -> list[str]:
        return self.git.changed_files

    @property
    def last_agent_meta(self) -> dict:
        return self.agent.last_agent_meta

    def scratch_files(self) -> list[str]:
        d = self.ws / ".scratch"
        if not d.is_dir():
            return []
        return sorted(str(p.relative_to(self.ws)) for p in d.rglob("*") if p.is_file())[:10]
