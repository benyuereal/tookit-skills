#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""explore —— 没有候选时，从**通用信号**出题；不拒绝，也不放松六道门。

## 为什么需要它（2026-11-27 实测）

C 臂在三臂测试里**唯一一次**执行 loop.py 是：

    loop.py -C Plugins/ops/skills/op-moe-topk-gating status
    → 候选  待做 0 / 已判死 0 / 停滞可作父代 0 / 共记录 0 条

然后它再也没碰过 loop.py，转身自己写 bench。全仓 16 个待做候选里，能映射到
**已实现变体**的只有 4 个。也就是说：引擎本身没坏，是"没活干"被 agent 正确地
读成了死胡同——这是引擎的失败，不是 agent 的。

## 本模块只做一件事

把「没活干」变成「有 N 条待实现的明确方向」，每条都说清：**哪个信号看到的、
指向哪个攻击面、为什么、怎么验收**。

## 它不绕过任何一道门

* 生成的候选 `status: proposed`，**不是** `pending`；
* `loop.py run` 仍然拒绝它（变体确实还不存在，冒名跑等于造假账）；
* `expect` 一律 `0%` + `expect_basis`，不许编一个百分比去搅乱 EV 排序；
* 方向**只允许**来自 `references/surfaces.yaml`，不许发明（`forge_next` 同规矩）。

## 信号从哪来：全部可推导，不需要先手写 roofline.json

`op-roofline/derive.py` 按算子模型直接算出内存/算力地板、`cta_per_cu`、
`lds_ratio`、`rounds`、`serial_chain`；再叠加历史 `runs/loop/*.jsonl` 的已达成
耗时和 `candidates.yaml` 的方向预算。少一样就少一条规则，绝不猜。

## 用法

    python3 scripts/explore.py --card Plugins/ops/skills/op-flash-mla            # 只看
    python3 scripts/explore.py --card Plugins/ops/skills/op-flash-mla --apply    # 写 overlay
    python3 scripts/explore.py --card ... --shape decode-m32 --json

`--apply` 写 `runs/advance/overlay.yaml`，`loop.py` 的 `load_candidates` 会合并它
并把这些候选标成 `_origin=advance-overlay`。**跑不跑仍由 loop.py 的六道门决定。**
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / ".git").exists()),
                 Path(__file__).resolve().parents[1])
SKILLS = REPO_ROOT / "Plugins/ops/skills"
DERIVE = SKILLS / "op-roofline/derive.py"
SURFACES = SKILLS / "op-optimize/references/surfaces.yaml"

# contract.yaml 的 op 名 → derive.py 的模型名。**没登记的算子就直接说没有模型，
# 不许拿一个"差不多的"模型凑数**（用 matmul 的模型去算 attention，地板能差 10 倍）。
DERIVE_MODEL = {
    "flash-mla": "sparse_mla",
    "sparse-mla": "sparse_mla",
    "mqa-logits": "mqa_logits",
    "gemm-bf16": "matmul",
    "gemm-fp8": "matmul",
}

# 语义化名字 → derive.py 的 --xxx 参数。derive.py 的参数名是算子特定的，
# 这里只做"从 shapes.yaml 的 params 里取哪些字段"的映射。
DERIVE_ARGS = {
    "sparse_mla": {"S": "--s", "H": "--h", "TOPK": "--topk", "D": "--d",
                   "SKV": "--skv", "BM": "--bm", "BI": "--bi"},
    "mqa_logits": {"S": "--s", "SKV": "--skv"},
    "matmul": {"M": "--m", "N": "--n", "K": "--k"},
}

# 方向白名单兜底：surfaces.yaml 读不到时用这份（与 2026-11-27 的仓库内容一致）。
SURFACES_FALLBACK = (
    "occupancy", "grid-tail", "prefetch-depth", "barrier-width", "layout-pack",
    "table-miss", "fusion-boundary", "precision-drop", "split-k", "gather-path",
    "cross-lane-fold", "slc-policy", "prefetch-depth-isa", "vmem-vector-width",
    "mls-path", "lds-read-form", "epilogue-expansion",
)

# 通用信号的达成率阈值。**这些是经验值，写在这里是为了可被审阅和改写**，
# 不是物理常数——所以每条规则输出里都带上用的哪个阈值。
BW_LOW = 0.60        # 低于这个：多半是多趟 HBM / 向量宽度不够
BW_MID = 0.85        # 高于这个：已经贴近地板，别再折腾 kernel 内部
SERIAL_RATIO = 1.30  # 串行地板 / 重叠地板 超过这个 = 并行度没吃到


def _yaml(path):
    import yaml
    p = Path(path)
    if not p.is_file():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def load_surfaces():
    d = _yaml(SURFACES)
    ss = d.get("surfaces") if isinstance(d, dict) else d
    out = [s.get("id") for s in ss or [] if isinstance(s, dict) and s.get("id")]
    return out or list(SURFACES_FALLBACK)


def card_shapes(card):
    d = _yaml(Path(card) / "shapes.yaml")
    return d if isinstance(d, dict) else {}


def contract_op(card):
    d = _yaml(Path(card) / "contract.yaml")
    return str(d.get("op") or ""), d


def run_derive(card, shape, timeout=180):
    """调 op-roofline/derive.py 拿地板与几何。跑不到就返回 (None, 原因)。

    **不在这里自己算地板**：地板是 op-roofline 的职责（它带 constants.yaml 的
    三层口径和模型），两处各算一遍必然对不上，而对不上的两个数会同时进报告。
    """
    op, _ = contract_op(card)
    model = DERIVE_MODEL.get(op)
    if not model:
        return None, "contract.yaml 的 op=%r 没有登记到 derive 模型（不猜）" % op
    if not DERIVE.is_file():
        return None, "找不到 %s" % DERIVE
    shp = card_shapes(card)
    if shape not in shp:
        return None, "shapes.yaml 里没有档位 %r（有：%s）" % (
            shape, ", ".join(sorted(shp)[:8]))
    params = (shp[shape] or {}).get("params") or {}
    args = [sys.executable, str(DERIVE), model, "--json"]
    mapping = DERIVE_ARGS.get(model) or {}
    for k, flag in mapping.items():
        if k in params:
            args += [flag, str(params[k])]
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, "derive.py 调不起来：%s: %s" % (type(e).__name__, e)
    if p.returncode != 0:
        return None, ("derive.py rc=%d：%s"
                      % (p.returncode, (p.stderr or p.stdout or "").strip()[:300]))
    try:
        return json.loads(p.stdout), None
    except json.JSONDecodeError as e:
        return None, "derive.py 输出不是 JSON：%s" % e


def achieved_us(card, shape=None):
    """从 runs/loop/*.jsonl 里取**已达到**的最小耗时（微秒）。没有就 None。

    只认 `measure_mode=event_isolated` 或未标注的记录：`back_to_back` 口径
    永远不可 claim（见 loop.py 的口径说明），拿它算达成率等于自己骗自己。
    """
    best = None
    for f in sorted((Path(card) / "runs" / "loop").glob("*.jsonl")):
        try:
            lines = f.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if shape and (d.get("shape") or {}).get("shape") not in (None, shape):
                continue
            if str(d.get("measure_mode")) not in ("event_isolated", "unknown", "None"):
                continue
            for k in ("cand_ms", "base_ms"):
                v = d.get(k)
                if isinstance(v, (int, float)) and v > 0:
                    us = float(v) * 1000.0
                    best = us if best is None else min(best, us)
    return best


def direction_budget(card):
    """每个方向还剩几次预算（2 次封方向）。返回 {direction: 剩余次数}。"""
    d = _yaml(Path(card) / "candidates.yaml")
    if isinstance(d, list):
        d = {"candidates": d}
    refutes = {}
    for c in d.get("candidates") or []:
        if not isinstance(c, dict):
            continue
        k = c.get("direction")
        if k:
            refutes.setdefault(k, 0)
            if str(c.get("status")) == "refuted":
                refutes[k] += 1
    return {k: int(d.get("direction_budget", 2)) - v for k, v in refutes.items()}


def available_variants(card):
    """这张卡现在**已实现**的变体名。三级回退，都拿不到就返回 (None, 原因)。

    为什么要三级：`_common.VARIANTS` 是最干净的，但实测 9 张卡里只有一部分导出它
    （op-flash-mla 就没有——它靠 `get_runner` 里的 `if name == "flash_mla"` 分派）。
    没有变体清单，"先量一次基线"这条候选就写不出来，而它恰恰是空白卡唯一的破局点。
    """
    import re
    card = Path(card)
    common = card / "_common.py"
    if common.is_file():
        src = common.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"^VARIANTS\s*=\s*\(([^)]*)\)", src, re.M)
        if m:
            got = [x.strip().strip("'\"") for x in m.group(1).split(",")]
            got = [x for x in got if x]
            if got:
                return got, "来自 _common.VARIANTS"
        # 启发式：从 get_runner 的分派分支里抽名字
        got = []
        for pat in (r'name\s*==\s*"([^"]+)"', r'name\s+in\s*\(([^)]*)\)'):
            for hit in re.finditer(pat, src):
                if hit.group(1):
                    got.extend(x.strip().strip("'\"") for x in hit.group(1).split(","))
        got = [g for g in dict.fromkeys(got) if g and g != "name"]
        if got:
            return got, "来自 _common.py 的 runner 分派分支（启发式）"
    ab = card / "bench" / "ab.py"
    if ab.is_file():
        m = re.search(r'DEFAULT_BASE\s*=\s*"([^"]+)"',
                      ab.read_text(encoding="utf-8", errors="replace"))
        if m:
            return [m.group(1)], "只拿到 bench/ab.py 的 DEFAULT_BASE"
    return None, "找不到已实现的变体清单（_common.VARIANTS / runner 分派 / ab.py 都没有）"


def collect_signals(card, shape):
    """把判断需要的通用事实集中取一次。取不到的项显式记 None + 原因。"""
    sig = {"card": Path(card).name, "shape": shape, "missing": []}
    roof, why = run_derive(card, shape)
    if roof is None:
        sig["missing"].append(why)
        sig["roof"] = None
    else:
        sig["roof"] = roof
    sig["surfaces"] = load_surfaces()
    sig["variants"], sig["variants_src"] = available_variants(card)
    sig["achieved_us"] = achieved_us(card, shape)
    sig["budget"] = direction_budget(card)

    if roof:
        fl = roof.get("floors_us") or {}
        geo = roof.get("geometry") or {}
        sig["floors_us"] = fl
        sig["bound"] = roof.get("cost_bound")
        sig["ceiling_us"] = roof.get("ceiling_us")
        sig["peak_gbs"] = roof.get("bandwidth_gbs")
        sig["ctas"] = geo.get("ctas")
        sig["cta_per_cu"] = geo.get("cta_per_cu")
        sig["rounds"] = geo.get("rounds")
        sig["lds_ratio"] = geo.get("lds_ratio")
        sig["lds_ok"] = geo.get("lds_ok")
        sig["serial_chain"] = geo.get("serial_chain")
        mem, comp = fl.get("memory"), fl.get("compute")
        overlap, serial = fl.get("overlap"), fl.get("serial")
        sig["serial_ratio"] = (serial / overlap) if overlap and serial else None
        sig["bound_gap"] = (min(mem, comp) / max(mem, comp)) if mem and comp else None
        ach, ceil = sig["achieved_us"], sig.get("ceiling_us")
        sig["efficiency"] = (ceil / ach) if (ach and ceil) else None
    return sig


def _mk(card, sig, n, direction, action_class, hypothesis, mechanism,
        expect_basis, todo, extra=None):
    """造一条**拟做**候选。status 一律 proposed —— 见模块说明。"""
    d = {
        "id": "auto-%s-%d" % (direction, n),
        "direction": direction,
        "status": "proposed",
        "priority": n,
        "action_class": action_class,
        "hypothesis": hypothesis,
        "expect": "0%",
        "expect_basis": expect_basis,
        "expect_confidence": "low",
        "note": mechanism,
        "todo": todo,
        "signal": {k: sig.get(k) for k in
                   ("bound", "efficiency", "cta_per_cu", "rounds", "lds_ratio",
                    "serial_ratio", "achieved_us", "ceiling_us")},
    }
    if extra:
        d.update(extra)
    return d


def propose(sig):
    """信号 → 候选。**只出方向，不出变体**；变体由人/agent 实现后改成 pending。

    每条规则都必须能回答三问：看到什么、指向哪个攻击面、怎么验收。
    """
    out, skipped = [], []
    surfs = set(sig.get("surfaces") or [])
    budget = sig.get("budget") or {}

    def allow(direction):
        """方向必须合法且还有预算。这两条任一不满足就不出——不是"少一条"，是"别提"。"""
        if direction not in surfs:
            skipped.append("%s：不在 surfaces.yaml 白名单" % direction)
            return False
        if budget.get(direction) is not None and budget[direction] <= 0:
            skipped.append("%s：方向预算已用完（2 次封方向）" % direction)
            return False
        return True

    n = 0

    def add(*a, **kw):
        nonlocal n
        n += 1
        out.append(_mk(sig, sig, n, *a, **kw))

    # ── 规则 0：没有可 claim 的实测 ⇒ 第一条候选就是"把基线量一次" ──────────
    # 这不是凑数。空白卡没候选⇒没跑过⇒没达成率⇒本该出的候选全出不来，是自锁。
    # 破它的正道不是猜一个方向，而是先测。引擎已经支持 purpose=calibration
    # （loop.py 会判 CALIBRATION_ONLY，不声称收益），这里只是把这条用起来。
    if not sig.get("achieved_us"):
        vs = sig.get("variants")
        if vs:
            add("occupancy", "measure-baseline",
                "这张卡在当前档位还没有可 claim 的实测（runs/loop 为空），"
                "所有基于达成率的判断都无从谈起；先把现成实现量一次档位基线",
                "用已有实现（%s）跑一次 event_isolated，拿到 achieved 与达成率，"
                "再让后续规则有数可用" % " / ".join(vs[:3]),
                "基线缺失，先校准；变体来自 %s，未做任何改动" % sig.get("variants_src"),
                ["把 status 改成 pending 并补 variant: %s" % vs[0],
                 "补 done_when（opcheck correctness）"],
                extra={"purpose": "calibration"})
        else:
            skipped.append("没有实测、也没有已实现变体可作基线——%s"
                           % (sig.get("variants_src") or ""))

    if not sig.get("roof"):
        skipped.append("没有 roofline 推导结果，除校准外的规则不启用：%s"
                       % "; ".join(sig.get("missing") or []))
        return out, skipped

    eff = sig.get("efficiency")
    bound = sig.get("bound")

    # ── 规则 1：串行地板远高于重叠地板 ⇒ 计算和数据搬运没重叠起来 ──────────
    sr = sig.get("serial_ratio")
    if sr and sr >= SERIAL_RATIO and allow("prefetch-depth"):
        add("prefetch-depth", "improve-overlap",
            "serial/overlap = %.2f ≥ %.2f，说明数据搬运与计算是串行的，"
            "吃到的是 serial 地板（%.1fµs）而不是 overlap 地板（%.1fµs）"
            % (sr, SERIAL_RATIO, sig["floors_us"].get("serial", 0),
               sig["floors_us"].get("overlap", 0)),
            "把下一块的 gather/load 提前到当前块的计算之前；" 
            "hw_kb 的 reg_resident_vs_lds 说明驻留方式决定能不能重叠",
            "由 derive 的 serial/overlap=%.2f 推出，尚未实测" % sr,
            ["实现一个 prefetch 变体", "补 done_when（opcheck correctness）"])

    # ── 规则 2：每个 CU 摊不到 1 个 CTA ⇒ 占用率不足 ───────────────────────
    cpu = sig.get("cta_per_cu")
    if isinstance(cpu, (int, float)) and cpu < 1 and allow("occupancy"):
        add("occupancy", "raise-parallelism",
            "cta_per_cu = %.2f < 1：CTA 总数 %s 填不满 CU，SM 有闲的"
            % (cpu, sig.get("ctas")),
            "把块切小 / 增加并行维度让 CTA 数 ≥ CU 数；warp_reduce_shfl 可用来说明"
            "跨 lane 归约不该再切块",
            "由 derive 的 geometry.cta_per_cu=%.2f 推出，尚未实测" % cpu,
            ["实现一个更细的切块变体", "补 done_when"])

    # ── 规则 3：grid 有多轮 ⇒ 尾轮不满 ─────────────────────────────────────
    r = sig.get("rounds")
    if isinstance(r, (int, float)) and r > 1 and allow("grid-tail"):
        add("grid-tail", "reduce-tail",
            "geometry.rounds = %s > 1：grid 要多轮才铺完，尾轮有 CTA 空转" % r,
            "让 grid 规模整除 CU 数，或把尾块合并进上一轮",
            "由 derive 的 geometry.rounds 推出，尚未实测",
            ["实现一个整除 CU 的切块", "补 done_when"])

    # ── 规则 4：LDS 预算吃紧 ⇒ num_stages 上不去 ───────────────────────────
    lr, lok = sig.get("lds_ratio"), sig.get("lds_ok")
    if ((isinstance(lr, (int, float)) and lr >= 0.80) or lok is False) and allow("lds-read-form"):
        add("lds-read-form", "reduce-lds",
            "LDS 占用比例 %s（lds_ok=%s）：shared memory 预算吃掉了流水深度"
            % (lr, lok),
            "改读形式省 LDS（例如直接读 VMEM 或复用寄存器）；"
            "reg_resident_vs_lds 里记着两条路线的取舍",
            "由 derive 的 geometry.lds_ratio=%s / lds_ok=%s 推出，尚未实测"
            % (lr, lok),
            ["实现一个省 LDS 的变体", "补 done_when"])

    # ── 规则 5/6/7：按达成率分档（memory bound 才有意义）───────────────────
    if bound == "memory" and isinstance(eff, (int, float)):
        if eff >= BW_MID:
            skipped.append("达成率 %.0f%% ≥ %.0f%%：已贴地板，不再出候选"
                           "（ceiling %.1fµs vs 实测 %.1fµs）"
                           % (eff * 100, BW_MID * 100,
                              sig.get("ceiling_us") or 0, sig.get("achieved_us") or 0))
        elif eff >= BW_LOW:
            for dr in ("slc-policy", "prefetch-depth"):
                if allow(dr):
                    add(dr, "improve-locality",
                        "memory bound，达成率只有 %.0f%%（%.1fµs vs 地板 %.1fµs），"
                        "差额多半在 L2/SLC 命中与预取深度上"
                        % (eff * 100, sig.get("achieved_us") or 0,
                           sig.get("ceiling_us") or 0),
                        "slc_cache_hint 与 vec16_uint4 说明了缓存驻留提示与向量宽度"
                        "各自能省多少往返",
                        "由达成率 %.0f%% 落在 [%.0f%%, %.0f%%) 推出，尚未实测"
                        % (eff * 100, BW_LOW * 100, BW_MID * 100),
                        ["实现一个 cache hint / 预取变体", "补 done_when"])
                    break
        else:
            for dr in ("vmem-vector-width", "layout-pack"):
                if allow(dr):
                    add(dr, "reduce-bytes",
                        "memory bound 但达成率只有 %.0f%%（%.1fµs vs 地板 %.1fµs）："
                        "典型原因是多趟 HBM 或向量宽度不够"
                        % (eff * 100, sig.get("achieved_us") or 0,
                           sig.get("ceiling_us") or 0),
                        "vec16_uint4 / store_width 说明了向量宽度对带宽的直接影响",
                        "由达成率 %.0f%% < %.0f%% 推出，尚未实测"
                        % (eff * 100, BW_LOW * 100),
                        ["实现一个向量化更宽的变体", "补 done_when"])
                    break
    elif bound == "memory":
        skipped.append("memory bound 但拿不到实测耗时（runs/loop 里没有可 claim 的记录），"
                       "算不出达成率 —— 先跑一次基线，别凭猜出候选")

    if not out:
        skipped.append("信号不足以支撑任何一条候选。这不是失败：说不出为什么的方向，"
                       "提出来只会浪费一次方向预算")
    return out, skipped


def render(sig, props, skipped):
    L = []
    L.append("=" * 78)
    L.append("explore — 从通用信号出题")
    L.append("=" * 78)
    roof = sig.get("roof") or {}
    L.append("卡片 %s   档位 %s" % (sig.get("card"), sig.get("shape")))
    if roof:
        fl = sig.get("floors_us") or {}
        L.append("地板  内存 %.1fµs / 算力 %.1fµs / 重叠 %.1fµs / 串行 %.1fµs  支配项=%s"
                 % (fl.get("memory", 0), fl.get("compute", 0),
                    fl.get("overlap", 0), fl.get("serial", 0), sig.get("bound")))
        L.append("几何  CTAs=%s  cta/cu=%s  rounds=%s  lds=%.2f  serial_chain=%s"
                 % (sig.get("ctas"), sig.get("cta_per_cu"), sig.get("rounds"),
                    sig.get("lds_ratio") or 0, sig.get("serial_chain")))
        L.append("达成  实测 %s  天花板 %.1fµs  达成率 %s"
                 % ("%.1fµs" % sig["achieved_us"] if sig.get("achieved_us") else "（无）",
                    sig.get("ceiling_us") or 0,
                    "%.0f%%" % (sig["efficiency"] * 100) if sig.get("efficiency") else "（算不出）"))
    for m in sig.get("missing") or []:
        L.append("缺  %s" % m)
    L.append("")
    if props:
        L.append("拟做 %d 条（**不是**待做：变体还没实现，loop.py run 会拒绝，这是对的）：" % len(props))
        for p in props:
            L.append("  [%d] %-24s direction=%-20s action=%s"
                     % (p["priority"], p["id"], p["direction"], p["action_class"]))
            L.append("      %s" % p["hypothesis"])
            L.append("      basis: %s" % p["expect_basis"])
            L.append("      todo : %s" % "；".join(p.get("todo") or []))
    else:
        L.append("拟做 0 条。")
    if skipped:
        L.append("")
        L.append("没出的原因：")
        for s in skipped:
            L.append("  - %s" % s)
    return "\n".join(L)


def explore(card, shape=None, limit=5):
    shp = card_shapes(card)
    if not shape:
        if not shp:
            return {"signals": {}, "proposals": [], "skipped": ["shapes.yaml 为空，无档位可选"]}
        # 默认档位：优先 decode 里工作量最大的（信号更容易暴露问题），其次第一个
        shape = sorted(shp, key=lambda k: (0 if k.startswith("decode") else 1, k))[0]
    sig = collect_signals(card, shape)
    props, skipped = propose(sig)
    return {"signals": sig, "proposals": props[:limit],
            "skipped": skipped + (props[limit:] and ["超出 limit %d，未列出" % limit] or [])}


def apply_overlay(card, props):
    """写进 runs/advance/overlay.yaml（复用 advance.write_overlay，不另造格式）。"""
    import importlib.util
    p = REPO_ROOT / "scripts" / "advance.py"
    spec = importlib.util.spec_from_file_location("_adv_for_overlay", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.write_overlay(card, props)


def main(argv=None):
    ap = argparse.ArgumentParser(description="从通用信号出候选（拟做，不是待做）")
    ap.add_argument("--card", required=True)
    ap.add_argument("--shape", default=None, help="shapes.yaml 的档位名；缺省自动选")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--apply", action="store_true", help="写 runs/advance/overlay.yaml")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    r = explore(a.card, a.shape, a.limit)
    if a.json:
        print(json.dumps(r, ensure_ascii=False, indent=2))
    else:
        print(render(r["signals"], r["proposals"], r["skipped"]))
    if a.apply:
        if not r["proposals"]:
            print("\n没有候选可写，不改 overlay。", file=sys.stderr)
            return 1
        f = apply_overlay(a.card, r["proposals"])
        print("\n已写 %s（%d 条，status=proposed）" % (f, len(r["proposals"])))
        print("下一步：实现变体 → 把 status 改成 pending 并补 variant/done_when → "
              "`loop.py -C %s run <id>`" % a.card)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
