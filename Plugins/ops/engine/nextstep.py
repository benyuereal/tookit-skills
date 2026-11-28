#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nextstep —— 「下一步做什么」的唯一策略实现。

## 为什么要有它（2026-11-28 的结论）

我们此前假设：把 loop.py 写进 CLAUDE.md 和 hook，agent 就会用它。
实测 C 臂（toolkit-skills）收到「本插件强制」+ 五条硬规则后，仍然把
「跑 loop.py」排在自己任务清单的第 4 位（前面是"自己建基准、自己出数"）。
连续三轮、三张卡，`runs/loop/*.jsonl` 都是 0 条。

原因不是它不会协调 —— 它自己建了 5 条任务清单并排序，协调能力强得很。
原因是**我们给的是动词（status/queue/run），不是决策边**：
从 `status` 的文字里推导"该跑哪个候选"要它自己合成策略，而合成我们的
策略比执行它自己的计划贵得多。理性选择就是绕开。

所以这一层要解决的是：**把"下一步"变成一个可以直接执行的结果**，
而不是一段需要解读的状态。Claude Code 仍然是协调者（它是运行时，
我们不可能替代它），但它每次只需要问一句：

    loop.py -C <卡> next

就能拿到 {action, why, cmd}，照做即可。策略留在引擎里，认知留给模型。

## 决策顺序（为什么是这个顺序）

1. **有实测却没有下一步** → 先看地板：贴地板就收工（继续抠是浪费预算）
2. **没有可 claim 的实测** → 先跑基线（calibration）。没有基线，
   后面任何"快了 x%"都没有分母 —— 这是六步循环的第 2 步，不能跳。
3. **有可跑候选** → 跑它（六道门会再筛一遍，这里只挑"有变体"的）
4. **有候选但变体没实现** → 实现它（这是唯一必须模型做的步骤）
5. **队列空** → 出题（explore.py 从通用信号生成「拟做」）

每一步都给出**可复制的命令**，不给"你应该考虑……"这种话。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

FLOOR_STOP = 0.10          # 剩余空间低于这个就别再抠（与 advance.AT_FLOOR 同源）
MAX_LISTED = 3


def _load(name):
    import importlib.util
    import sys
    key = "_ns_" + name
    if key in sys.modules:
        return sys.modules[key]
    p = Path(__file__).resolve().parent / (name + ".py")
    spec = importlib.util.spec_from_file_location(key, str(p))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


def implementable_variants(card: Path) -> set:
    """这张卡现在**能跑**的变体名。

    三条来源都要认（踩过：只认 VARIANTS 会把 op-fused-add-rmsnorm 的
    `fa-rms-mr2` 当不存在，因为它在 HIP_VARIANTS 里）：
      * `_common.VARIANTS`            —— 常规变体
      * `_common.HIP_VARIANTS`        —— 手写 HIP，按 so 文件名登记
      * `runs/forged/*.py`            —— 现场生成的变体
    """
    common = card / "_common.py"
    got = set()
    if common.is_file():
        src = common.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"^VARIANTS\s*=\s*\(([^)]*)\)", src, re.M)
        if m:
            got |= {x.strip().strip("'\"") for x in m.group(1).split(",") if x.strip()}
        # 不收尾匹配「\n}」：单行字典会漏（实测测试夹具就是单行）。
        # 字典值里不含 }，所以第一个 } 就是收尾。
        m2 = re.search(r"^HIP_VARIANTS\s*=\s*\{(.*?)\}", src, re.M | re.S)
        if m2:
            got |= set(re.findall(r'"([^"]+)"\s*:', m2.group(1)))
    forged = card / "runs" / "forged"
    if forged.is_dir():
        got |= {p.stem for p in forged.glob("*.py")}
    return got


def read_records(card: Path) -> list:
    out = []
    d = card / "runs" / "loop"
    if not d.is_dir():
        return out
    for f in sorted(d.glob("*.jsonl")):
        try:
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    out.append(json.loads(line))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def claimable(rec: dict) -> bool:
    """这条记录算不算"可 claim 的实测"。

    只认 event_isolated 口径、且真的形成了决定。`back_to_back` 永远不可 claim
    （见 loop.py 的口径说明）；`INCONCLUSIVE` 是噪声，不构成基线。
    """
    if str(rec.get("measure_mode")) != "event_isolated":
        return False
    dec = str(rec.get("decision") or "")
    return dec.startswith("ACCEPTED") or dec == "CALIBRATION_ONLY"


def floor_state(card: Path):
    """剩余空间。取不到就返回 None —— 不猜。"""
    try:
        f = _load("floor")
        fl = f.resolve(card)
        if getattr(fl, "ok", False) and isinstance(getattr(fl, "remaining", None),
                                                   (int, float)):
            return float(fl.remaining), getattr(fl, "source", None)
    except Exception:                            # noqa: BLE001
        return None
    return None


def deliverable_args() -> str:
    """如果当前目录像"题目工作目录"（有 results/），返回 `--bench-std` 片段。

    这条是为了让 agent **一条命令同时拿到账本和交付物**，不必回头研究
    `loop.py` 源码才知道怎么交差（实测：它真的回头读了 120 行源码）。
    """
    try:
        if (Path.cwd() / "results").is_dir():
            return " --bench-std results/bench_std.json"
    except OSError:
        pass
    return ""


def plan_next(card_dir, limit: int = MAX_LISTED) -> dict:
    """算出下一步。返回 {action, why, cmd, detail}，action ∈

        run        跑一个已有变体的候选
        implement  实现一个还没有变体的候选（唯一必须模型做的步骤）
        explore    队列空了，从通用信号出题
        baseline   还没有可 claim 的实测，先跑基线
        stop       已贴地板或已收敛，别再做
    """
    card = Path(card_dir).resolve()
    cid = card.name
    cardio = _load("card")
    cands = cardio.load_candidates(card)
    recs = read_records(card)
    variants = implementable_variants(card)

    pending = [c for c in cands.get("candidates") or []
               if isinstance(c, dict) and str(c.get("status", "pending")) == "pending"]
    proposed = [c for c in cands.get("candidates") or []
                if isinstance(c, dict) and str(c.get("status")) == "proposed"]
    refuted = cands.get("refuted") or []
    calib = [r for r in recs if claimable(r)]

    base = {"card": cid, "n_records": len(recs), "n_claimable": len(calib),
            "n_pending": len(pending), "n_proposed": len(proposed),
            "n_refuted": len(refuted)}

    def mk(action, why, cmd, **kw):
        d = dict(base, action=action, why=why, cmd=cmd, detail=kw)
        return d

    # ① 已有实测 → 看地板
    if calib:
        fs = floor_state(card)
        if fs is not None:
            remaining, src = fs
            base["remaining"] = remaining
            base["floor_source"] = src
            if remaining < FLOOR_STOP:
                return mk("stop",
                          "剩余空间 %.3f < %.2f（地板来源 %s）：继续抠的收益低于测量噪声"
                          % (remaining, FLOOR_STOP, src),
                          "python3 -I Plugins/ops/engine/loop.py -C %s status" % card)

    # ② 有可跑候选 → 直接跑。**跑出来的第一条记录本身就是基线**，
    #    不需要先"补一条 calibration 再跑"（那会把能跑的东西降级成准备工作）。
    runnable = [c for c in pending if c.get("variant") in variants]
    if runnable:
        runnable.sort(key=lambda c: c.get("priority", 99))
        c = runnable[0]
        more = [x.get("id") for x in runnable[1:limit]]
        _why = ("有 %d 条候选的变体已实现且可跑；先跑优先级最高的那条"
                "（跑完引擎会自己判定并记账，不需要你再解释结果）" % len(runnable))
        if not calib:
            _why = ("还没有任何可 claim 的实测，而这条候选的变体已经实现 —— "
                    "直接跑它，第一条记录就是基线")
        _rounds = int((cands.get("defaults") or {}).get("rounds") or 3)
        return mk("run", _why,
                  "python3 -I Plugins/ops/engine/loop.py -C %s run %s%s"
                  % (card, c.get("id"), deliverable_args()),
                  tag=c.get("id"), variant=c.get("variant"), also=more,
                  expect_seconds="约 %d-%d 秒（%d 轮 × event 计时；"
                                 "若这张卡的变体要现场编译 HIP，首次可能 2-5 分钟）"
                                 % (5 * _rounds, 30 * _rounds, _rounds),
                  produces=["`%s/runs/loop/<日期>.jsonl` 追加一条记录"
                            "（含判定：ACCEPTED_*/REJECTED_*/CALIBRATION_ONLY）" % card.name,
                            "`results/bench_std.json`（命令里带了 --bench-std 时）",
                            "`op-atlas/cards/<id>.yaml` 的 t_actual_us 被写回"],
                  on_failure=["rc=2：六道门拒绝了 —— 看输出最后三行，它会说"
                              "是哪道门（方向预算/瓶颈归因/done_when/天花板）",
                              "rc=1：A/B 失败 —— 通常是变体缺编译产物或库 import 失败，"
                              "看 `variant=... 跳过: ...` 那行",
                              "任何失败都**不要自己手搓测量**：跑完再 `next` 一次，"
                              "它会给出新的下一步"])

    # ③ 候选在、变体没实现 → 实现它（唯一必须模型做的步骤）
    todo = [c for c in pending if c.get("variant") not in variants]
    if todo:
        todo.sort(key=lambda c: c.get("priority", 99))
        c = todo[0]
        want = c.get("variant") or ("auto-" + str(c.get("id")))
        return mk("implement",
                  "候选 %r 已登记但变体 %r 不可用 —— 这是唯一必须你来做的步骤："
                  "把它实现出来并登记进 _common.VARIANTS，之后 loop.py 接管度量与判定"
                  % (c.get("id"), want),
                  "编辑 %s/_common.py：把 %r 加进 VARIANTS 并给 get_runner 加分支"
                  % (card, want),
                  tag=c.get("id"), variant=want,
                  spec={"direction": c.get("direction"), "expect": c.get("expect"),
                        "hypothesis": c.get("hypothesis") or c.get("note")})

    # ④ 只有「拟做」 → 挑一条实现
    if proposed:
        ids = [c.get("id") for c in proposed[:limit]]
        return mk("implement",
                  "队列里只有 %d 条「拟做」（还没有变体）；挑一条实现出来" % len(proposed),
                  "编辑 %s/_common.py：实现 %s 并把 status 改成 pending + 补 variant"
                  % (card, ", ".join(str(x) for x in ids)),
                  candidates=ids)

    # ⑤ 全空 → 出题
    return mk("explore",
              "队列空了（待做 0、拟做 0）。从通用信号（roofline/带宽/占用率/LDS）出题，"
              "生成的是「拟做」——还要你实现才能跑，这不是死胡同",
              "python3 -I scripts/explore.py --card %s --apply" % card)


def render(plan: dict) -> str:
    """人读渲染。**第一行必须是下一步**，因为 agent 常常只读一行就行动。"""
    act = plan.get("action")
    label = {"run": "跑实验", "implement": "实现变体", "explore": "出题",
             "baseline": "先测基线", "stop": "停止"}.get(act, act)
    L = ["下一步：%s（%s）" % (label, act), ""]
    L.append("为什么：%s" % plan.get("why", ""))
    L.append("")
    L.append("照做（复制粘贴即可）：")
    L.append("    %s" % plan.get("cmd", ""))
    d = plan.get("detail") or {}
    if d.get("expect_seconds"):
        L.append("")
        L.append("预计耗时：%s" % d["expect_seconds"])
    if d.get("produces"):
        L.append("")
        L.append("跑完之后你会得到：")
        for x in d["produces"]:
            L.append("  · %s" % x)
    if d.get("on_failure"):
        L.append("")
        L.append("如果失败：")
        for x in d["on_failure"]:
            L.append("  · %s" % x)
    if d.get("also"):
        L.append("")
        L.append("之后还有：%s" % ", ".join(str(x) for x in d["also"]))
    if d.get("spec"):
        L.append("")
        L.append("这条候选的说明：")
        for k, v in d["spec"].items():
            if v:
                L.append("  %-12s %s" % (k, str(v)[:160]))
    L.append("")
    L.append("当前状态：记录 %d 条（可 claim %d）/ 待做 %d / 拟做 %d / 已判死 %d%s"
             % (plan.get("n_records", 0), plan.get("n_claimable", 0),
                plan.get("n_pending", 0), plan.get("n_proposed", 0),
                plan.get("n_refuted", 0),
                ("  remaining=%s" % plan["remaining"]) if "remaining" in plan else ""))
    return "\n".join(L)
