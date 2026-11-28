# -*- coding: utf-8 -*-
"""signals —— 采信号：把「判断需要哪些通用事实」集中取一次。

## 为什么单独一个模块

`scripts/explore.py` 原来的 600 行里，前面 330 行全是**采数**（跑 derive 模型、
读 runs/loop 的已达成耗时、扫 _common.py 找已实现变体、数方向预算、
读 surfaces 白名单），后面才是「信号 → 候选」的规则。两者混在一个文件里的
后果是：想知道「这张卡为什么没出候选」，只能把整个文件读一遍。

拆开之后职责变成一句话：
  · **probe**：取事实。取不到的项显式记 None + 原因，**绝不猜一个数**。
  · **propose**：消费事实，产出候选。每条都必须能回答「看到什么 / 指向哪个
    攻击面 / 为什么 / 怎么验收」。

## 纪律

本模块不产出候选、不做判定。它唯一允许的"判断"是「这个信号有没有取到」。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / ".git").exists()),
                 Path(__file__).resolve().parents[1])

_ENGINE_DIR = Path(__file__).resolve().parent


def _engine(name):
    """按路径加载 engine/ 下的兄弟模块（见 engine/loop.py::_engine 的说明）。"""
    key = "_ops_engine_" + name
    if key in sys.modules:
        return sys.modules[key]
    import importlib.util
    spec = importlib.util.spec_from_file_location(key, str(_ENGINE_DIR / (name + ".py")))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod


card = _engine("card")
floor = _engine("floor")

SKILLS = REPO_ROOT / "Plugins/ops/skills"
SURFACES = SKILLS / "op-optimize/references/surfaces.yaml"

# 方向白名单兜底：surfaces.yaml 读不到时用这份（与 2026-11-27 的仓库内容一致）。
# 为什么需要兜底而不是直接报错：这张表是**知识**，卡片的正确性不该依赖它是否
# 可读；但表在的时候必须以表为准（见 load_surfaces 的返回）。
SURFACES_FALLBACK = (
    "occupancy", "grid-tail", "prefetch-depth", "barrier-width", "layout-pack",
    "table-miss", "fusion-boundary", "precision-drop", "split-k", "gather-path",
    "cross-lane-fold", "slc-policy", "prefetch-depth-isa", "vmem-vector-width",
    "mls-path", "lds-read-form", "epilogue-expansion",
)


def load_surfaces():
    """可用的攻击面 id 列表。表读不到才用兜底（兜底也会被打印出来）。"""
    d = card.read_yaml(SURFACES, {})
    ss = d.get("surfaces") if isinstance(d, dict) else d
    out = [s.get("id") for s in ss or [] if isinstance(s, dict) and s.get("id")]
    return out or list(SURFACES_FALLBACK)


def achieved_us(card_dir, shape=None):
    """从 runs/loop/*.jsonl 里取**已达到**的最小耗时（微秒）。没有就 None。

    只认 `measure_mode=event_isolated` 或未标注的记录：`back_to_back` 口径
    永远不可 claim（见 loop.py 的口径说明），拿它算达成率等于自己骗自己。
    """
    best = None
    for d in card.load_roofline_measurements(card_dir):
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


def direction_budget(card_dir):
    """每个方向还剩几次预算（2 次封方向）。返回 {direction: 剩余次数}。

    注意这里数的是 `candidates[].status == refuted`，与 loop.py 的
    `direction_table` 数 `directions[].refutes` 是两个来源。这个不一致是
    **既有的**（见组件报告），本模块只做只读统计，不在这里悄悄改口径。
    """
    d = card.load_candidates(card_dir)
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


def available_variants(card_dir):
    """这张卡现在**已实现**的变体名。三级回退，都拿不到就返回 (None, 原因)。

    为什么要三级：`_common.VARIANTS` 是最干净的，但实测 9 张卡里只有一部分导出它
    （op-flash-mla 就没有——它靠 `get_runner` 里的 `if name == "flash_mla"` 分派）。
    没有变体清单，"先量一次基线"这条候选就写不出来，而它恰恰是空白卡唯一的破局点。
    """
    card_dir = Path(card_dir)
    common = card_dir / "_common.py"
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
    ab = card_dir / "bench" / "ab.py"
    if ab.is_file():
        m = re.search(r'DEFAULT_BASE\s*=\s*"([^"]+)"',
                      ab.read_text(encoding="utf-8", errors="replace"))
        if m:
            return [m.group(1)], "只拿到 bench/ab.py 的 DEFAULT_BASE"
    return None, "找不到已实现的变体清单（_common.VARIANTS / runner 分派 / ab.py 都没有）"


def collect(card_dir, shape):
    """把判断需要的通用事实集中取一次。取不到的项显式记 None + 原因。

    返回的 dict 就是 `explore --json` 里的 `signals`，字段名是**对外契约**
    （测试与 agent 都在读 `signals.roof` / `signals.achieved_us` / `signals.missing`）。

    地板只在这里取一次（`floor.resolve`），不再「先试 derive、失败再起子进程
    调 floor.py」—— 那个顺序现在是 engine/floor.py 里的唯一实现。
    """
    card_dir = Path(card_dir)
    sig = {"card": card_dir.name, "shape": shape, "missing": []}
    fl = floor.resolve(card_dir, shape, sources=("derive", "atlas"))
    # `roof` 保留 derive 的**原始 JSON**：propose 要用几何量（ctas/rounds/lds/
    # serial_chain），Floor 只给了统一视图。
    roof = fl.detail if fl.source == "derive" else None
    sig["roof"] = roof
    # 输出的来源标签沿用旧名：`derive.py` = 算子模型，`floor.py` = 按访问模式算的
    # 通用地板（现在的实现在 engine/floor.py 的 atlas 分支）。改字会打断既有的
    # 对账脚本与验收基线，而这两个名字指的是**层**，不是文件名。
    sig["floor_src"] = "derive.py" if roof else None
    if roof is None:
        if fl.source == "atlas":
            d = fl.detail
            sig["floor_src"] = "floor.py"
            sig["bound"] = d.get("binding")
            sig["ceiling_us"] = d.get("t_floor_us")
            sig["achieved_us"] = d.get("t_actual_us")
            sig["efficiency"] = d.get("utilization")
            sig["gate"] = d.get("gate")
            sig["remaining"] = d.get("remaining")
            sig["floor_advice"] = d.get("advice")
            sig["achieved_gbs"] = d.get("achieved_gbs")
            sig["peak_gbs"] = d.get("denominator_gbs")
            sig["floor_basis"] = d.get("basis")
        else:
            # 一个来源都没取到：把「哪个来源、为什么」逐条摊开。
            sig["missing"].extend(fl.missing)
    sig["surfaces"] = load_surfaces()
    sig["variants"], sig["variants_src"] = available_variants(card_dir)
    # **不要用 runs/loop 的最小值覆盖 floor 给的 t_actual**：floor 的达成率
    # 分母是 atlas 卡的 t_actual_us，两者必须是同一次测量，否则同一句话里
    # 会出现"达成率 18%（617.4µs vs 地板 116.4µs）"这种自相矛盾的输出
    # （116.4/617.4=18.9%，116.4/629.6=18.5%）。实测 2026-11-28。
    _best_record = achieved_us(card_dir, shape)
    sig["best_record_us"] = _best_record
    if sig.get("achieved_us") is None:
        sig["achieved_us"] = _best_record
        sig["achieved_from"] = "runs/loop"
    else:
        sig["achieved_from"] = "atlas"
    sig["budget"] = direction_budget(card_dir)

    if roof:
        fl_us = roof.get("floors_us") or {}
        geo = roof.get("geometry") or {}
        sig["floors_us"] = fl_us
        sig["bound"] = roof.get("cost_bound")
        sig["ceiling_us"] = roof.get("ceiling_us")
        sig["peak_gbs"] = roof.get("bandwidth_gbs")
        sig["ctas"] = geo.get("ctas")
        sig["cta_per_cu"] = geo.get("cta_per_cu")
        sig["rounds"] = geo.get("rounds")
        sig["lds_ratio"] = geo.get("lds_ratio")
        sig["lds_ok"] = geo.get("lds_ok")
        sig["serial_chain"] = geo.get("serial_chain")
        mem, comp = fl_us.get("memory"), fl_us.get("compute")
        overlap, serial = fl_us.get("overlap"), fl_us.get("serial")
        sig["serial_ratio"] = (serial / overlap) if overlap and serial else None
        sig["bound_gap"] = (min(mem, comp) / max(mem, comp)) if mem and comp else None
        ach, ceil = sig["achieved_us"], sig.get("ceiling_us")
        sig["efficiency"] = (ceil / ach) if (ach and ceil) else None
        rem = geo.get("rounds")  # 只为保持字段存在，真正的剩余空间由 floor.py 给
        sig.setdefault("gate", None if rem is None else sig.get("gate"))
    return sig


def pick_shape(card_dir, shape=None):
    """没给档位时挑一个：优先 decode 里工作量最大的（信号更容易暴露问题），
    其次第一个。shapes.yaml 为空返回 None。"""
    shp = card.load_shapes(card_dir)
    if shape:
        return shape
    if not shp:
        return None
    return sorted(shp, key=lambda k: (0 if k.startswith("decode") else 1, k))[0]
