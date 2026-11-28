#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""coordinate —— **运行时协调者**：由它决定跑什么，agent 只负责实现。

## 为什么必须有它（2026-11-28 读 C 臂会话得出的结论）

我们一直假设「把 loop.py 写进 CLAUDE.md + 加一条 hook，agent 就会用」。
实测相反：C 臂收到的是「本插件强制」+ 五条硬规则，它照做的是——

    #2 Build GPU-event benchmark with L2 flush
    #3 Run measurement and write bench_std.json     ← 自己出数
    #4 Run loop.py on the op-fused-add-rmsnorm card ← 引擎在最后，打卡

也就是说 **agent 自己当了协调者**：它建任务清单、自己排序、把引擎排到第四位。
我们那三层（静态文档 / 运行时提示 / 事后校验）全是被动的——文档能被重新解释，
hook 进了上下文也只是"参考信息"，校验只能扣分不能改路径。

结论：**协调者必须在 agent 外面。** 本文件就是那个协调者：

    循环 { 读队列 → 选候选 → （没有可跑变体时）叫 agent 实现 → loop.py run
           → 读判定 → 记账 → 再规划 } 直到预算/收敛/无候选

agent 只是它调用的一个"实现服务"；判断、记账、推进全在这里和 loop.py。

## 与 advance.py 的分工

`advance.py` 做的是「计划 → 门 → 跑 → 判 → 记录 → 重规划」的**骨架**，
但它假设候选的变体**已经实现好了**。本文件补的正是缺的那一环：
**把"实现变体"变成一次可调用的步骤**（`Implementer`），从而真正闭环。

## 闭环的验收（这就是「tookit-skills 闭环」的定义）

一次 `coordinate --max-rounds N` 跑完，必须同时满足：
  1. 产生了至少 N 条 `runs/loop/*.jsonl` 记录（引擎实测的账本，不是自报）；
  2. 每轮都有决定（ACCEPTED_*/REJECTED_*/CALIBRATION_ONLY），不是 INCONCLUSIVE 空转；
  3. 起点的队列与终点的队列不同（有推进：status 变了或有新记录）；
  4. 全程没有人敲命令（除了启动这一条）。

## 纪律

* **不绕过 loop.py 的六道门**：所有实验都通过子进程调 `loop.py run`，本文件不自己测。
* **不编造实现**：implementer 说实现了，就**验证变体能被 `_common.VARIANTS` 解析**
  （`loop.py` 会拒），验不过就记 `impl_failed` 并放弃该候选，绝不硬跑。
* 每一步都落 `runs/coordinate/*.jsonl`，可事后逐条对账。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = next((p for p in Path(__file__).resolve().parents if (p / ".git").exists()),
            Path(__file__).resolve().parents[1])
ENGINE = REPO / "Plugins/ops/engine"
LOOP = ENGINE / "loop.py"

sys.path.insert(0, str(ENGINE))
sys.path.insert(0, str(REPO / "scripts"))


# ---------------------------------------------------------------- 工具

def _load_engine(name):
    """按路径加载 engine 模块（本仓没有 packaging，`-I` 又会踢掉 sys.path）。"""
    import importlib.util
    key = "_coord_" + name
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, str(ENGINE / (name + ".py")))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def emit(msg):
    print(msg, flush=True)


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def coord_dir(card: Path) -> Path:
    d = card / "runs" / "coordinate"
    d.mkdir(parents=True, exist_ok=True)
    return d


def log_step(card: Path, rec: dict) -> None:
    f = coord_dir(card) / (time.strftime("%Y%m%d") + ".jsonl")
    with f.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_ledger(card: Path) -> list:
    out = []
    for f in sorted((card / "runs" / "loop").glob("*.jsonl")):
        try:
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    out.append(json.loads(line))
        except (OSError, json.JSONDecodeError):
            continue
    return out


# ---------------------------------------------------------------- 状态读取

def snapshot(card: Path) -> dict:
    """读一次卡的当前状态。**只读，不判定** —— 判定是 loop.py 的事。"""
    cardio = _load_engine("card")
    cands = cardio.load_candidates(card)
    ledger = read_ledger(card)
    pending, proposed, blocked = [], [], []
    for c in cands.get("candidates") or []:
        if not isinstance(c, dict):
            continue
        st = str(c.get("status", "pending"))
        if st == "pending":
            pending.append(c)
        elif st == "proposed":
            proposed.append(c)
        elif st == "blocked":
            blocked.append(c)
    return {
        "card": card.name,
        "pending": pending,
        "proposed": proposed,
        "blocked": blocked,
        "n_refuted": len(cands.get("refuted") or []),
        "n_records": len(ledger),
        "records": ledger,
        "defaults": cands.get("defaults") or {},
    }


def runnable(card: Path, snap: dict) -> list:
    """能直接跑的候选：pending + 声明的 variant 真的在 `_common.VARIANTS` 里。

    **不做额外判定**：方向预算、瓶颈归因、天花板这些门由 `loop.py run` 拦。
    这里只回答"它有没有变体可跑"，因为那是 implementer 该补的洞。
    """
    common = card / "_common.py"
    variants = set()
    if common.is_file():
        import re
        src = common.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"^VARIANTS\s*=\s*\(([^)]*)\)", src, re.M)
        if m:
            variants = {x.strip().strip("'\"") for x in m.group(1).split(",") if x.strip()}
        # **手写 HIP 变体不在 VARIANTS 里**：它们登记在 HIP_VARIANTS（名字 → so 文件）。
        # loop.py 的 get_runner 也有一条注释写着同一件事（"要在 VARIANTS 检查之前"）。
        # 这里不补的话，`op-fused-add-rmsnorm` 的 reg-resident 会被误判成
        # "没有可跑变体"——实测 2026-11-28，协调者当场报"队列空了"。
        m2 = re.search(r"^HIP_VARIANTS\s*=\s*\{(.*?)\n\}", src, re.M | re.S)
        if m2:
            variants |= set(re.findall(r'"([^"]+)"\s*:', m2.group(1)))
        # 现场生成的变体落在 runs/forged/*.py，get_runner 也会认
        forged = card / "runs" / "forged"
        if forged.is_dir():
            variants |= {p.stem for p in forged.glob("*.py")}
    out = []
    for c in snap["pending"]:
        v = c.get("variant")
        if v and (not variants or v in variants):
            out.append(c)
    return sorted(out, key=lambda c: c.get("priority", 99))


# ---------------------------------------------------------------- 实现变体

class Implementer:
    """把「一个候选」变成「一个可跑变体」。子类负责具体怎么造。"""

    name = "base"

    def can(self, card: Path, cand: dict) -> bool:
        raise NotImplementedError

    def implement(self, card: Path, cand: dict) -> tuple[bool, str]:
        raise NotImplementedError


class NoopImplementer(Implementer):
    """不造变体。用于验证控制流（以及声明"这一步本该由 agent 做"）。"""
    name = "none"

    def can(self, card, cand):
        return False

    def implement(self, card, cand):
        return False, "none"


class AgentImplementer(Implementer):
    """叫一个 Claude Code 子进程去实现变体 —— **agent 在这里是服务，不是主导**。

    为什么用子进程而不是让本进程 import：agent 要能改文件、跑编译、自己验证，
    那是它的强项；协调者只要求它交付一个可验证的结果（变体名能被解析）。

    权限：`--dangerously-skip-permissions` 在 root 下被 Claude Code 拒绝
    （实测 2026-11-28），所以走 `su - <user>`。工作副本必须属于该用户。
    """
    name = "agent"

    def __init__(self, user="opb_c", env_file="/workspace/tsoc.env",
                 timeout=900, model_flag=None):
        self.user = user
        self.env_file = env_file
        self.timeout = timeout
        self.model_flag = model_flag

    def can(self, card, cand):
        return True

    def _prompt(self, card: Path, cand: dict, variant: str, work_card: Path) -> str:
        return f"""你是一个算子实现服务。只做一件事：把下面的候选实现成 loop.py 能跑的变体。

卡目录：{work_card}
（这是可写副本。**只改这个目录里的文件**；仓库里的原件由协调者拷回。）
候选 id：{cand.get('id')}
方向：{cand.get('direction')}
预期：{cand.get('expect')}
机制说明：{" ".join(str(cand.get('note') or cand.get('means') or '').split())[:600]}

要交付的变体名（必须一字不差）：{variant}

硬要求：
1. 在 `{work_card}/_common.py` 的 `VARIANTS` 元组里加入 "{variant}"；
2. 在 `get_runner(name, **kwargs)` 里加一个分支，返回该变体的可调用对象；
3. 该可调用对象的输入输出语义必须和 `reference()` 完全一致（同一个数学）；
4. 自检：`cd {work_card} && python3 -c "import _common; assert '{variant}' in _common.VARIANTS; print('OK')"`；
5. **只改这一个卡目录里的文件**。不要改 contract.yaml / shapes.yaml / 其它卡 / 任何 scripts。

完成后只输出一行：DONE {variant}
如果做不到，只输出一行：FAILED <一句话原因>
"""

    def implement(self, card: Path, cand: dict) -> tuple[bool, str]:
        import shutil
        variant = "auto-" + str(cand.get("id") or "x")
        ws = Path("/tmp/coordinator") / ("ws-" + card.name)
        if ws.exists():
            shutil.rmtree(ws, ignore_errors=True)
        ws.mkdir(parents=True, exist_ok=True)
        # 父目录必须 o+x：实测它是 root 700，opb_c 连穿越都不行，
        # 报出来却是"PROMPT.txt: Permission denied"，很容易误判成文件权限。
        os.chmod(ws.parent, 0o755)
        os.chmod(ws, 0o755)
        work_card = ws / card.name
        # 1) 整张卡拷进可写工作区（agent 要读 contract/shapes 才知道语义）
        shutil.copytree(card, work_card,
                        ignore=shutil.ignore_patterns("runs", "__pycache__", "*.so"))
        subprocess.run(["chown", "-R", "%s:%s" % (self.user, self.user), str(ws)],
                       check=False, capture_output=True)

        prompt_file = ws / "PROMPT.txt"
        prompt_file.write_text(self._prompt(card, cand, variant, work_card),
                               encoding="utf-8")
        out_file = ws / "agent.out"
        script = ws / "run_agent.sh"
        script.write_text(
            "#!/bin/sh\n"
            "set -a; . %s; set +a\n"
            "ANTHROPIC_BASE_URL=${ANTHROPIC_BASE_URL%%/}\n"
            "cd %s\n"
            "exec %s -p --output-format json --allowedTools Read Write Edit Glob Grep Bash "
            "--dangerously-skip-permissions\n"
            % (self.env_file, ws, "/opt/opbench-node/bin/claude"),
            encoding="utf-8")
        script.chmod(0o755)
        subprocess.run(["chown", "-R", "%s:%s" % (self.user, self.user), str(ws)],
                       check=False, capture_output=True)

        cmd = "su - %s -c 'sh %s < %s' > %s 2>&1" % (
            self.user, script, prompt_file, out_file)
        try:
            subprocess.run(cmd, shell=True, timeout=self.timeout, check=False)
        except subprocess.TimeoutExpired:
            return False, "agent 超时（%ds）" % self.timeout
        txt = out_file.read_text(encoding="utf-8", errors="replace") if out_file.exists() else ""
        if "DONE" not in txt:
            return False, "agent 没交付（输出尾部：%s）" % " ".join(txt.split())[-400:]

        # 2) 只拷回实现文件：_common.py + kernels/ 下的新文件
        src_common = work_card / "_common.py"
        if not src_common.is_file():
            return False, "工作区里没有 _common.py"
        new_src = src_common.read_text(encoding="utf-8", errors="replace")
        if variant not in new_src:
            return False, "agent 说 DONE 但 _common.py 里没有 %r" % variant
        (card / "_common.py").write_text(new_src, encoding="utf-8")
        ksrc, kdst = work_card / "kernels", card / "kernels"
        if ksrc.is_dir():
            kdst.mkdir(parents=True, exist_ok=True)
            for f in ksrc.iterdir():
                if f.is_file() and f.suffix in (".hip", ".cu", ".cpp", ".py", ".h", ".cuh"):
                    shutil.copy2(f, kdst / f.name)

        # 3) **验证，不信自述**：变体名必须真能被 _common 解析，
        #    且 get_runner 真的要得出来（有的实现只在 VARIANTS 里写了名字）。
        probe = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, %r); import _common; "
             "assert %r in _common.VARIANTS, 'not in VARIANTS: %%r' %% (_common.VARIANTS,); "
             "r = _common.get_runner(%r); assert callable(r), 'runner 不可调用'; "
             "print('OK')" % (str(card), variant, variant)],
            capture_output=True, text=True, cwd=str(card))
        if probe.returncode != 0:
            return False, ("变体解析/取 runner 失败：%s"
                           % ((probe.stderr or probe.stdout or "").strip()[-300:]))
        return True, "变体 %s 已实现，VARIANTS 与 get_runner 双向验证通过" % variant


# ---------------------------------------------------------------- 跑一轮

def run_loop(card: Path, tag: str, bench_std: Path | None = None,
             timeout=1800) -> tuple[int, str]:
    cmd = [sys.executable, "-I", str(LOOP), "-C", str(card), "run", tag]
    if bench_std:
        cmd += ["--bench-std", str(bench_std)]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 124, "loop.py run 超时（%ds）" % timeout
    return p.returncode, (p.stdout or "") + (("\n[stderr]\n" + p.stderr) if p.stderr else "")


def latest_decision(card: Path) -> dict | None:
    recs = read_ledger(card)
    return recs[-1] if recs else None


def coordinate(card: Path, max_rounds: int, implementer: Implementer,
               dry_run=False, bench_std=None, emit_fn=emit) -> dict:
    """主循环。返回报告 dict（也落 runs/coordinate）。"""
    card = Path(card).resolve()
    report = {"card": card.name, "ts": now(), "rounds": 0, "steps": [],
              "status": "done", "reason": "", "implementer": implementer.name,
              "start_records": len(read_ledger(card))}
    emit_fn("=" * 78)
    emit_fn("coordinate — 由协调者决定跑什么，agent 只负责实现")
    emit_fn("卡 %s   最大轮数 %d   实现方式 %s%s"
            % (card.name, max_rounds, implementer.name, "（dry-run）" if dry_run else ""))
    emit_fn("=" * 78)

    for rnd in range(1, max_rounds + 1):
        snap = snapshot(card)
        emit_fn("\n[第 %d 轮] 待做 %d / 拟做 %d / 已判死 %d / 已有记录 %d"
                % (rnd, len(snap["pending"]), len(snap["proposed"]),
                   snap["n_refuted"], snap["n_records"]))
        cands = runnable(card, snap)
        if not cands:
            # 没有可跑变体：让 implementer 从 pending/proposed 里挑一个去实现
            runnable_ids = {c.get("id") for c in cands}
            pool = ([c for c in snap["pending"] if c.get("id") not in runnable_ids]
                    or snap["proposed"])
            if not pool:
                report.update(status="stopped", reason="没有候选可推进（pending/proposed 都为空）")
                emit_fn("  停：队列空了。下一步该是 explore.py 出题。")
                break
            cand = pool[0]
            if not implementer.can(card, cand):
                report.update(status="stopped",
                              reason="无可跑变体，且实现方式=%s 不接管" % implementer.name)
                emit_fn("  停：候选 %r 没有可跑变体，实现方式 %r 不接管。"
                        % (cand.get("id"), implementer.name))
                break
            emit_fn("  候选 %r 没有可跑变体 → 交给实现服务" % cand.get("id"))
            if dry_run:
                emit_fn("  [dry-run] 跳过实现与实测")
                report["steps"].append({"round": rnd, "action": "implement", "dry_run": True})
                break
            ok, why = implementer.implement(card, cand)
            log_step(card, {"ts": now(), "round": rnd, "action": "implement",
                            "tag": cand.get("id"), "ok": ok, "why": why})
            report["steps"].append({"round": rnd, "action": "implement", "ok": ok, "why": why})
            emit_fn("  %s %s" % ("✓" if ok else "✗", why))
            if not ok:
                report.update(status="blocked", reason="实现失败：%s" % why)
                break
            snap = snapshot(card)
            cands = runnable(card, snap)
            if not cands:
                report.update(status="blocked", reason="实现了但变体仍不可跑")
                break

        cand = cands[0]
        tag = cand.get("id")
        emit_fn("  跑候选 %r（variant=%r）" % (tag, cand.get("variant")))
        if dry_run:
            emit_fn("  [dry-run] 不调 loop.py")
            report["steps"].append({"round": rnd, "action": "run", "tag": tag, "dry_run": True})
            break
        before = len(read_ledger(card))
        rc, out = run_loop(card, tag, bench_std=bench_std)
        emit_fn("  loop.py run rc=%d" % rc)
        for line in out.strip().splitlines()[-6:]:
            emit_fn("    " + line)
        rec = latest_decision(card)
        after = len(read_ledger(card))
        ok = rc == 0 and after > before
        log_step(card, {"ts": now(), "round": rnd, "action": "run", "tag": tag,
                        "rc": rc, "new_records": after - before,
                        "decision": (rec or {}).get("decision"),
                        "delta_pct": (rec or {}).get("delta_pct")})
        report["steps"].append({"round": rnd, "action": "run", "tag": tag, "rc": rc,
                                "new_records": after - before,
                                "decision": (rec or {}).get("decision")})
        report["rounds"] = rnd
        if not ok:
            report.update(status="blocked", reason="loop.py 拒绝了 %r（rc=%d）" % (tag, rc))
            emit_fn("  停：引擎拒绝了这个候选（rc=%d）。这不是协调者的失败——"
                    "六道门拦住了不该跑的东西。" % rc)
            break
        emit_fn("  判定 %s" % (rec or {}).get("decision"))

    if report["status"] == "done" and report["rounds"] >= max_rounds:
        report["reason"] = "达到轮数上限"
    report["end_records"] = len(read_ledger(card))
    card_ok = report["end_records"] > report["start_records"]
    report["closed_loop"] = bool(card_ok and report["rounds"] > 0)
    log_step(card, {"ts": now(), "action": "summary", **{k: report[k] for k in
                                                         ("rounds", "status", "reason",
                                                          "closed_loop")}})
    emit_fn("\n" + "=" * 78)
    emit_fn("结论：%s  轮数 %d  账本 %d→%d  闭环=%s"
            % (report["status"], report["rounds"], report["start_records"],
               report["end_records"], "是" if report["closed_loop"] else "否"))
    if report["reason"]:
        emit_fn("原因：" + report["reason"])
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description="运行时协调者：决定跑什么，agent 只负责实现")
    ap.add_argument("--card", required=True)
    ap.add_argument("--max-rounds", type=int, default=3, dest="max_rounds")
    ap.add_argument("--implementer", default="none",
                    choices=["none", "agent"],
                    help="没有可跑变体时谁来造：none=只报告，agent=叫 Claude Code 子进程")
    ap.add_argument("--user", default="opb_c", help="agent 以哪个用户跑（root 下 skip-permissions 被拒）")
    ap.add_argument("--bench-std", default=None, help="让 loop.py 顺带写交付物")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    impl = AgentImplementer(user=a.user) if a.implementer == "agent" else NoopImplementer()
    rep = coordinate(Path(a.card), a.max_rounds, impl, dry_run=a.dry_run,
                     bench_std=a.bench_std)
    if a.json:
        print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0 if rep["closed_loop"] or a.dry_run else 1


if __name__ == "__main__":
    sys.exit(main())
