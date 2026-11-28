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

# 建议命令里的脚本一律用绝对路径：调用者的 cwd 可能是题目工作目录，
# 不是仓库根 —— 相对路径会给出**一条跑不起来的命令**（实测 2026-11-28）。
_LOOP_PY = Path(__file__).resolve().parent / "loop.py"
_REPO = Path(__file__).resolve().parents[3]

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
        # 题目工作目录的形状：有 results/（opbench 约定）。此时同时产出
        # 交付物与报告 —— 一条命令交差，引擎才不比"自己干活"贵。
        if (Path.cwd() / "results").is_dir():
            return " --bench-std results/bench_std.json --report report.md"
    except OSError:
        pass
    return ""


def _smallest_shape(card: Path) -> tuple[str, dict]:
    """取 shapes.yaml 里元素数最小的那档，smoke 用（不碰大形状）。"""
    import re as _re
    p = card / "shapes.yaml"
    if not p.is_file():
        return "", {}
    try:
        import sys as _sys
        _sys.path.insert(0, str(_REPO / "scripts"))
        from opverify import parse_yaml
        data = parse_yaml(p.read_text(encoding="utf-8")) or {}
    except Exception:                                        # noqa: BLE001
        return "", {}
    best, best_n = "", None
    for k, v in data.items():
        params = dict((v or {}).get("params") or {})
        n = 1
        for x in params.values():
            if isinstance(x, int) and x > 0:
                n *= x
        if best_n is None or n < best_n:
            best, best_n = k, n
    return best, dict((data.get(best) or {}).get("params") or {})


def smoke_adapter(card: Path, variant: str) -> tuple[list, list]:
    """**真的把 adapter 跑一次**（CPU、最小形状），返回 (checks, problems)。

    为什么必须真跑：`implementable_variants` 只看名字在不在 VARIANTS 里。
    新建卡（op_bootstrap）的 `build_inputs()` / `reference()` 是按设计留的
    NotImplementedError 占位符，于是 `next` 会给出"直接执行不会失败"的
    自信结论，而第一次 `run` 必然炸在 build_inputs（实测 2026-11-28）。
    模型据此会认为引擎坏了，转回手搓测量 —— 闭环就断在这里。
    """
    checks, problems = [], []
    common = card / "_common.py"
    if not common.is_file():
        return checks, ["缺 _common.py"]
    import importlib.util
    import sys as _sys
    tag = "_smoke_%s" % abs(hash(str(card)))
    spec = importlib.util.spec_from_file_location(tag, str(common))
    m = importlib.util.module_from_spec(spec)
    _sys.path.insert(0, str(card))
    try:
        spec.loader.exec_module(m)
    except Exception as e:                                   # noqa: BLE001
        return checks, ["_common.py 都 import 不了：%s: %s" % (type(e).__name__, e)]
    if not hasattr(m, "build_inputs"):
        # 老卡/测试夹具可能只有 VARIANTS + get_runner，没有 build_inputs。
        # 这时不做 smoke（没有可验证的输入构造），保持向后兼容。
        return checks, []
    _key, params = _smallest_shape(card)
    try:
        try:
            inp = m.build_inputs(params, seed=0, device="cpu")
        except TypeError:
            inp = m.build_inputs(params, seed=0)
        checks.append("build_inputs() 可执行（形状 %s，CPU）" % (_key or "?"))
    except NotImplementedError:
        return checks, ["`build_inputs()` 还是 bootstrap 的占位符 —— "
                        "这是 op-optimize 六步的第 2 步，必须先按真实分布实现"]
    except Exception as e:                                   # noqa: BLE001
        return checks, ["build_inputs() 跑不起来：%s: %s" % (type(e).__name__, e)]
    if not variant or variant == "ref-heavy":
        try:
            runner = m.get_runner(variant, **(m.runner_kwargs(params)
                                             if hasattr(m, "runner_kwargs") else {}))
            args = inp[:-1] if isinstance(inp[-1], dict) else inp
            runner(*args)
            checks.append("reference/get_runner(%r) 可执行" % variant)
        except NotImplementedError:
            problems.append("`reference()` 还是 bootstrap 的占位符 —— "
                            "先补真实数学（正确性的唯一锚），再跑实验")
        except Exception as e:                               # noqa: BLE001
            problems.append("reference/get_runner(%r) 跑不起来：%s: %s"
                            % (variant, type(e).__name__, e))
    else:
        checks.append("库变体 %r 未做 CPU smoke（需要 GPU/后端）" % variant)
    return checks, problems


def preflight(card: Path, cand: dict) -> dict:
    """这一条候选**能不能直接跑**。返回 {"ok": bool, "checks": [...], "problems": [...]}。

    复用 loop.py 自己的六道门函数 —— 不在这里另写一套判据（那正是本仓反复
    在消灭的"两份口径"）。跑在 1 秒内，不碰 GPU。
    """
    checks, problems = [], []
    v = cand.get("variant")
    vs = implementable_variants(card)
    if v and v in vs:
        checks.append("变体 %r 已实现（VARIANTS / HIP_VARIANTS / forged 三处之一）" % v)
    else:
        problems.append("变体 %r 不在可跑清单里（当前可跑：%s）"
                        % (v, ", ".join(sorted(vs)[:6]) or "无"))
    bench = card / (cand.get("_bench") or "bench/ab.py")
    if bench.is_file():
        checks.append("适配器 %s 存在" % bench.name)
    else:
        problems.append("缺适配器 %s" % bench)
    # 真的把 adapter 跑一次：名字在 VARIANTS 里 ≠ 能跑（占位符 reference
    # 会在这里被抓出来，而不是等第一次 run 炸掉）。
    try:
        _c, _p = smoke_adapter(card, str(v or ""))
        checks.extend(_c)
        problems.extend(_p)
    except Exception as e:                                   # noqa: BLE001
        problems.append("adapter smoke 自身失败：%s: %s" % (type(e).__name__, e))
    try:
        loop = _load("loop")
        cands = _load("card").load_candidates(card)
        for fn, label in (("bottleneck_problems", "瓶颈归因"),
                          ("done_when_problems", "done_when"),
                          ("budget_problems", "方向预算")):
            f = getattr(loop, fn, None)
            if f is None:
                continue
            try:
                kw = {"tags": {cand.get("id")}}
                if fn == "bottleneck_problems":
                    kw["card"] = card
                probs = f(cands, **kw)
            except TypeError:
                probs = f(cands)
            if probs:
                problems.extend("%s：%s" % (label, p) for p in probs[:2])
            else:
                checks.append("六道门·%s 通过" % label)
    except Exception as e:                       # noqa: BLE001
        problems.append("门禁自检跑不起来（%s: %s）—— 不能保证会跑通" % (type(e).__name__, e))
    return {"ok": not problems, "checks": checks, "problems": problems}


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
                          "python3 -I %s -C %s status" % (_LOOP_PY, card))

    # ② 有可跑候选 → 直接跑。**跑出来的第一条记录本身就是基线**，
    #    不需要先"补一条 calibration 再跑"（那会把能跑的东西降级成准备工作）。
    # 已经跑过（账本里有同一 variant 的记录）的候选不再重复跑 ——
    # 否则校准候选永远 pending，`next` 会无限次让你 run 同一条（实测 2026-11-28）。
    done_variants = {str(r.get("tag")) for r in recs if r.get("tag")}
    runnable = [c for c in pending
                if c.get("variant") in variants
                and str(c.get("variant")) not in done_variants]
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
        _pf = preflight(card, c)
        if not _pf["ok"]:
            # 预检不过就别让它去试 —— 直接把问题写出来，省一轮失败。
            # 占位符 reference/build_inputs 是**第 2 步没做**，不是"跑之前
            # 再检查一下"，必须单独说清楚，否则模型会以为引擎坏了。
            if any("占位符" in p for p in _pf["problems"]):
                return mk("implement",
                          "这张卡还没完成 op-optimize 的**第 2 步**："
                          "`build_inputs()` / `reference()` 还是 bootstrap 的占位符。"
                          "把它们补成真实实现，引擎才有可信的测量与判据。",
                          "编辑 %s/_common.py：实现 build_inputs() 与 reference()"
                          % card,
                          tag=c.get("id"), variant=c.get("variant"),
                          problems=_pf["problems"])
            return mk("fix",
                      "这条候选还不能跑，预检发现 %d 个问题（见下）。先照修，"
                      "修完再 next" % len(_pf["problems"]),
                      "看下面的 problems 逐条修",
                      tag=c.get("id"), variant=c.get("variant"),
                      problems=_pf["problems"])
        return mk("run", _why,
                  "python3 -I %s -C %s run %s%s"
                  % (_LOOP_PY, card, c.get("id"), deliverable_args()),
                  tag=c.get("id"), variant=c.get("variant"), also=more,
                  expect_seconds="约 %d-%d 秒（%d 轮 × event 计时；"
                                 "若这张卡的变体要现场编译 HIP，首次可能 2-5 分钟）"
                                 % (5 * _rounds, 30 * _rounds, _rounds),
                  produces=["`%s/runs/loop/<日期>.jsonl` 追加一条记录"
                            "（含判定：ACCEPTED_*/REJECTED_*/CALIBRATION_ONLY）" % card.name,
                            "`results/bench_std.json`（命令里带了 --bench-std 时）",
                            "`op-atlas/cards/<id>.yaml` 的 t_actual_us 被写回"],
                  self_checks=_pf["checks"],
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
              "python3 -I %s --card %s --apply" % (_REPO / "scripts/explore.py", card))


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
    if d.get("self_checks"):
        L.append("")
        L.append("已自检（1 秒，未跑实验）——直接执行不会失败：")
        for x in d["self_checks"]:
            L.append("  ✓ %s" % x)
    if d.get("problems"):
        L.append("")
        L.append("预检不通过（先修这些）：")
        for x in d["problems"]:
            L.append("  ✗ %s" % x)
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
