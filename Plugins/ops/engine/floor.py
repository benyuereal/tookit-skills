#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""floor —— 「还能优化多少」只能有一个数，且只由一处算出来。

## 为什么收口：三份实现会给出三个地板

2026-11-27 实测，算「地板 / 剩余空间」这件事在仓库里被实现了三遍，各读各的：

  · `scripts/advance.py::read_roofline()`        —— 只读 `<卡>/roofline.json`
  · `scripts/floor.py::card_spec()`              —— 只读 `op-atlas/cards/<id>.yaml`
                                                   的 `floor_inputs`
  · `scripts/explore.py::run_derive()` /
    `floor_via_script()`                         —— 先试 `op-roofline/derive.py`，
                                                   失败再**起子进程**调 floor.py

同一个算子、同一时刻，三处能给出不同的地板，而三个数都会进报告。
「能不能再挖」这种判据出现三个数，等于没有判据。

## 取数顺序（唯一）

  ① **derive** —— `op-roofline/derive.py` 的算子模型。最准，因为它给的不只是
     地板，还有 cta_per_cu / rounds / lds_ratio / serial_chain 这些几何量。
     代价：要 contract.yaml 的 op 登记过模型，且必须给出档位形状。
  ② **roofline** —— `<卡>/roofline.json`，derive 的离线产物。没有形状时的降级。
  ③ **atlas** —— `op-atlas/cards/<id>.yaml` 的 `floor_inputs`，按访存模式的
     通用地板。任何算子都能走通（只要 `bytes_mb` + `pattern` + `t_actual_us`）。

`sources=` 让调用方声明它**信哪几层**，因为不是所有消费方都该吃 atlas 兜底 ——
见 `scripts/advance.py` 的 at_floor 前置门：它只信 ①②。

## 不抛异常，也不猜

取不到就返回一个 `t_floor_us is None` 的 Floor，把缺什么写进 `missing`
（"缺 roofline.json" / "contract.yaml 的 op=x 没登记模型" / "缺 floor_inputs"……）。
**故意不返回 None**：返回 None 时 missing 就没地方挂了，而「为什么算不出」
恰恰是这类工具最需要说清的事。用 `floor.ok` 判有没有数。

## ③ 的算法是搬过来的，不是重写的

`analyze()` 及其配套（带宽阶梯、两条腿取慢、launch 加性项、归约折扣、gate 分档）
原本就在 `scripts/floor.py` 里，且带一整套 selftest 兜着。这次只搬位置，
口径一个字没动 —— 重构不许顺手改判据。
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / ".git").exists()),
                 Path(__file__).resolve().parents[3])

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


_CARD = _engine("card")

SKILLS = REPO_ROOT / "Plugins/ops/skills"
DERIVE = SKILLS / "op-roofline/derive.py"
CONSTANTS = SKILLS / "op-roofline/constants.yaml"
ATLAS_CARDS = REPO_ROOT / "Plugins/ops/op-atlas/cards"
LADDER = REPO_ROOT / "Plugins/ops/op-atlas/bw-ladder.yaml"

# 三个来源的**默认优先级**。顺序即语义，别在调用点重排。
SOURCES = ("derive", "roofline", "atlas")

# 模式 → bw-ladder.yaml 的层键
PATTERN_TO_LAYER = {
    "r_only": "2_read_only",
    "w_only": "2_write_only",
    "rw_1to1": "2_rw_1to1",
    "rw_2to1": "2_rw_2to1",
    "cross_row": "3_cross_row",
    "scatter_gather": "3_scatter_gather",
}

# (阈值, gate 名, 建议)。按阈值从大到小匹配：remaining >= 阈值 就采用该档。
GATE = ((0.30, "headroom", "先怀疑基线/语义/天花板错了，不是先挖"),
        (0.10, "structural", "做结构变换（fusion / split / layout）"),
        (0.0, "at_floor", "at_floor，别再折腾"))

# 算力峰值：只认 achievable 口径（见 constants.yaml 三层口径）。
# impl_measured 不许当分母——那是某一条实现的值。
COMPUTE_PEAK = {
    "bf16": ("compute", "bf16_gemm_peak_tf"),
    "fp8": ("compute", "fp8_peak_layers"),
}

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


# ------------------------------------------------------------------ 统一结果

@dataclass
class Floor:
    """一个地板的**全部**可回答信息。

    字段刻意分三组，因为它们的**可信度不同**，混在一起就会有人拿
    `t_actual_us is None` 去当「这个算子没优化空间」：

      来源组   source / bound / basis / geometry / detail —— 这个数从哪来
      地板组   t_floor_us —— 理论最短时间（µs）
      账目组   t_actual_us / efficiency / remaining / gate —— 实测与差距

    `efficiency` = t_floor / t_actual（达成率，1.0 = 贴着地板）。
    `remaining`  = t_actual / t_floor − 1（还剩多少空间，负数 = 判据错了）。
    两个数方向相反，历史上被混用过（explore 里就同时用着两处的值），
    所以这里把名字写全，谁用哪个一看就懂。
    """
    source: str = None                 # "derive" / "roofline" / "atlas" / None
    bound: str = None                  # compute / memory / mixed
    t_floor_us: float = None
    t_actual_us: float = None
    efficiency: float = None           # t_floor / t_actual
    remaining: float = None            # t_actual / t_floor - 1
    gate: str = None                   # headroom / structural / at_floor / below_floor
    basis: str = ""                    # 人读的依据（分母、档位、折扣……）
    geometry: dict = field(default_factory=dict)   # ctas/cta_per_cu/rounds/lds/…
    missing: list = field(default_factory=list)    # 缺什么（算不出时非空）
    inputs: dict = field(default_factory=dict)     # 算之前归一化的输入（可复算）
    detail: dict = field(default_factory=dict)     # 来源的原始产物（JSON 可比对）

    @property
    def ok(self):
        """有没有算出地板。**只看这一个**，别用 `t_actual_us` 判断。"""
        return self.t_floor_us is not None

    def as_json(self):
        """稳定的机器可读形状（CLI --json / 报告用）。"""
        return {
            "source": self.source, "bound": self.bound,
            "t_floor_us": self.t_floor_us, "t_actual_us": self.t_actual_us,
            "efficiency": self.efficiency, "remaining": self.remaining,
            "gate": self.gate, "basis": self.basis,
            "geometry": self.geometry, "missing": list(self.missing),
        }


def _num(v):
    """数字或 None。bool 要挡住：YAML 里 `launch_us: false` 会变成 0/1。"""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def _fill_accounting(f, t_floor=None, t_actual=None, efficiency=None, remaining=None):
    """把 t_actual / efficiency / remaining / gate 补齐（能补多少补多少）。

    为什么集中在一处：达成率与剩余空间是**两个方向相反的量**，历史上
    explore 与 floor 各算一遍、混着用（explore 拿 floor 的 utilization 配
    自己的 achieved，得到「18% 但 617µs」这种自相矛盾的行）。口径只有一份。

    已经算好的 `efficiency` / `remaining` 会被原样采用而**不重算**：
    `analyze()` 里的值是拿未取整的地板算的，而 `Floor.t_floor_us` 用的是
    取整到 0.1µs 的数。重算会让 `--json` 的末位变（4.4101 → 4.4089），
    这类「重构后数变了」的 diff 最难解释，也最容易掩盖真问题。
    """
    if t_floor is not None:
        f.t_floor_us = t_floor
    if t_actual is not None:
        f.t_actual_us = t_actual
    if remaining is not None:
        f.remaining = remaining
    if efficiency is not None:
        f.efficiency = efficiency
    if f.remaining is None and f.t_floor_us and f.t_actual_us:
        f.remaining = round(f.t_actual_us / f.t_floor_us - 1.0, 4)
    if f.efficiency is None and f.t_floor_us and f.t_actual_us:
        f.efficiency = round(f.t_floor_us / f.t_actual_us, 4)
    if f.gate is None and f.remaining is not None:
        f.gate = gate_of(f.remaining)[0]
    return f


# ------------------------------------------------------------------ ③ atlas 地板

def load_ladder():
    """读带宽阶梯。读不到就**停下**：地板的分母不可猜（1724 那次就是猜的）。"""
    if not LADDER.is_file():
        raise SystemExit("缺少 %s —— 先跑 python3 -I scripts/bw_calibrate.py" % LADDER)
    d = _CARD.read_yaml(LADDER, {})
    if not isinstance(d, dict) or not d:
        raise SystemExit("%s 读不出内容（应为 yaml dict）" % LADDER)
    return d


def load_peak(precision):
    """读 constants.yaml 的 achievable 算力峰值。取不到就报错，不猜。"""
    if not CONSTANTS.is_file():
        raise SystemExit("缺少 %s" % CONSTANTS)
    c = _CARD.read_yaml(CONSTANTS, {})
    node = (c.get("compute") or {}).get(COMPUTE_PEAK[precision][1])
    if not isinstance(node, dict):
        raise SystemExit("constants.yaml 缺 compute.%s" % COMPUTE_PEAK[precision][1])
    if precision == "fp8":
        v = node.get("achievable_tf")
    else:
        v = node.get("value")
    if not v:
        raise SystemExit("constants.yaml 的 %s 没有 achievable 值——"
                         "impl_measured 不许当分母" % COMPUTE_PEAK[precision][1])
    return float(v)


def ladder_value(ladder, pattern):
    key = PATTERN_TO_LAYER.get(pattern)
    if not key:
        return None, "未知访存模式 %r（可选 %s）" % (pattern, sorted(PATTERN_TO_LAYER))
    node = (ladder.get("layers") or {}).get(key) or {}
    val = node.get("achievable")
    if not val:
        return None, "该模式未标定：%s" % (node.get("reason") or "无记录")
    return float(val), "confidence=%s" % node.get("confidence", "?")


def gate_of(remaining):
    """返回 (判据, 建议)。remaining 可为 None（算不出）。"""
    if remaining is None:
        return "unknown", "缺数据，无法判定"
    if remaining < 0:
        return "below_floor", "已在地板以下 —— 说明上限估低了或流量算多了，**先查这个**"
    for th, name, advice in GATE:
        if remaining >= th:
            return name, advice
    return "at_floor", "at_floor，别再折腾"


def reduction_penalty(ladder, flag):
    """归约类算子的带宽折扣。实测 rmsnorm 三实现 0.75~0.81，取 0.79。

    不折扣会算出 +51% 这种成熟实现不可能达到的假空间——
    2026-11-23 rmsnorm-fused 实测：我们 242.2 / lightop 197.4 / tilelang 204.5，
    按纯 copy 地板 159.8 us 算分别是 +51.6% / +23.5% / +28.0%。
    lightop 是厂商实现，不可能离地板 23%，说明纯 copy 档对归约算子取高了。
    """
    if not flag:
        return 1.0, ""
    node = (ladder.get("layers") or {}).get("3_row_reduction") or {}
    r = node.get("penalty_ratio")
    if not r:
        return 1.0, "（归约惩罚未标定，地板可能偏高）"
    return float(r), "；归约折扣 %.2f（bw-ladder 3_row_reduction）" % float(r)


def memory_floor(bytes_mb, pattern, ladder):
    peak, note = ladder_value(ladder, pattern)
    if not peak:
        return None, None, note
    return bytes_mb / peak * 1000.0, peak, "%s 档 %.0f GB/s（%s）" % (pattern, peak, note)


def compute_floor(flops, peak_tf):
    return flops / (peak_tf * 1e12) * 1e6, "%.1f TF achievable" % peak_tf


def analyze(spec, ladder):
    """spec 字段：bytes_mb / pattern / flops / peak_tf(或 precision) / t_actual_us。

    **两个地板都算，取更慢的那个当 binding。** 只算一个会给出方向相反的结论。

    launch_us 是加性项：小 kernel 的耗时下界是「launch + 数据搬运」，
    不是「数据搬运」。不加上它，topk（34.2 µs vs 43 µs launch 地板）会算出
    负 remaining——那是判据本身错了，不是实现突破了物理。
    """
    t = spec.get("t_actual_us")
    launch = float(spec.get("launch_us") or 0.0)
    floors = []          # (时间 µs, 名称, 依据, 分母)
    if spec.get("bytes_mb") and spec.get("pattern"):
        f, peak, why = memory_floor(spec["bytes_mb"], spec["pattern"], ladder)
        if f is None:                       # 模式未标定：先返回错误，别再往下算
            return {"error": why}
        pen, pen_why = reduction_penalty(ladder, spec.get("is_reduction"))
        f = f / pen                         # 归约类地板按折扣放宽（时间变长 = 上限降低）
        why += pen_why
        floors.append((f, "memory", why, peak))
    if spec.get("flops"):
        peak_tf = spec.get("peak_tf") or load_peak(spec.get("precision") or "bf16")
        f, why = compute_floor(spec["flops"], peak_tf)
        floors.append((f, "compute", why, peak_tf))
    if not floors:
        return {"error": "既没有 bytes_mb+pattern，也没有 flops —— 给不出地板"}

    bind = max(floors, key=lambda x: x[0])
    t_floor, bind_kind, bind_why, bind_den = bind
    t_floor += launch                       # launch 是加性固定代价，不是可重叠项
    remaining = (t / t_floor - 1.0) if t else None
    g, advice = gate_of(remaining)

    if len(floors) == 2:
        other = min(floors, key=lambda x: x[0])
        note_other = "另一条腿 %s 地板 %.1f us（%.2fx 更松）" % (
            other[1], other[0], t_floor / other[0])
    else:
        note_other = ""
    parts = ["%s 地板 %.1f us（%s）" % (k, f, w) for f, k, w, _ in floors]
    basis = "；".join(parts) + "；binding=%s" % bind_kind
    if launch:
        basis += "；+ launch %.0f us" % launch
    if t:
        basis += "；实测 %.1f us" % t
    if note_other:
        basis += "；" + note_other
    # E40 的输入：这个实现实际跑出来的带宽/算力。
    # 它们物理上不可能超过本机 achievable——超过只可能是分母或字节算错了。
    achieved_gbs = (spec["bytes_mb"] / t * 1000.0) if (t and spec.get("bytes_mb")) else None
    achieved_tf = (spec["flops"] / t / 1e6) if (t and spec.get("flops")) else None
    return {
        "pattern": spec.get("pattern"),
        "achieved_gbs": round(achieved_gbs, 1) if achieved_gbs else None,
        "achieved_tf": round(achieved_tf, 1) if achieved_tf else None,
        "denominator_gbs": bind_den if bind_kind == "memory" else None,
        "denominator_tf": bind_den if bind_kind == "compute" else None,
        "denominator_source": ("Plugins/ops/op-atlas/bw-ladder.yaml" if bind_kind == "memory"
                               else "Plugins/ops/skills/op-roofline/constants.yaml"),
        "bytes_mb": spec.get("bytes_mb"),
        "flops": spec.get("flops"),
        "binding": bind_kind,
        "basis": basis,
        "t_floor_us": round(t_floor, 1),
        "t_actual_us": t,
        "remaining": round(remaining, 4) if remaining is not None else None,
        "utilization": round(t_floor / t, 4) if t else None,
        "gate": g,
        "advice": advice,
    }


def atlas_spec(card_id):
    """从 atlas 卡的 `floor_inputs` 里取输入。缺字段就明确说缺什么，不猜。

    返回 (spec, None) 或 (None, 原因)。`floor.py --card` 的 SKIP 行用的就是
    这个原因串 —— 它是给人看的第一手线索，别改成 "N/A"。
    """
    p = ATLAS_CARDS / ("%s.yaml" % card_id)
    if not p.is_file():
        raise SystemExit("找不到卡 %s" % p)
    d = _CARD.read_yaml(p, {})
    fi = d.get("floor_inputs")
    if not isinstance(fi, dict):
        return None, "缺 floor_inputs（要有 bytes_mb/pattern 或 flops，以及 t_actual_us/basis）"
    spec = {k: fi.get(k) for k in
            ("bytes_mb", "pattern", "flops", "peak_tf", "precision",
             "t_actual_us", "is_reduction", "launch_us", "floor_override",
             "t_source", "evidence")}
    if not fi.get("basis"):
        return None, "floor_inputs 缺 basis —— 只有数字没有依据，拒算"
    # 业主 2026-11-24 裁决③：反推的 t 单列，不进 measured 统计。
    # derived = t 由「字节 ÷ 卡里记的 GB/s」推出来的，不是直接读到的时间；
    # missing = 连算都算不出（缺单一形状的字节或耗时）。二者都拒算 remaining。
    src = fi.get("t_source") or "results"
    # 只排除两类：derived（由 GB/s × 字节反推的 t）和 missing（连 t 都没有）。
    # results / card / constants 都是「读过一次真实测量的时间」，照常计入。
    # 第一版把 card 也挡了，结果 rmsnorm-fused 这种有 40 次轮换中位的卡被误伤。
    if src in ("derived", "missing"):
        return None, "t_source=%s —— 单列不进统计：%s" % (
            src, fi.get("missing_evidence") or "t 不是直接实测的时间")
    if spec["t_actual_us"] is None:
        return None, "floor_inputs 缺 t_actual_us —— 没有实测耗时就没有 remaining"
    spec["basis"] = fi["basis"]
    spec["t_source"] = src
    spec["card"] = card_id
    return spec, None


# ------------------------------------------------------------------ ① derive 地板

def run_derive(card, shape, timeout=180):
    """调 op-roofline/derive.py 拿地板与几何。跑不到就返回 (None, 原因)。

    **不在这里自己算地板**：地板是 op-roofline 的职责（它带 constants.yaml 的
    三层口径和模型），两处各算一遍必然对不上，而对不上的两个数会同时进报告。
    """
    if not shape:
        return None, "没给档位形状：derive 模型按形状算，不给形状就没有模型可用"
    contract = _CARD.load_contract(card)
    op = str(contract.get("op") or "")
    model = DERIVE_MODEL.get(op)
    if not model:
        return None, "contract.yaml 的 op=%r 没有登记到 derive 模型（不猜）" % op
    if not DERIVE.is_file():
        return None, "找不到 %s" % DERIVE
    shp = _CARD.load_shapes(card)
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
    import json
    try:
        return json.loads(p.stdout), None
    except json.JSONDecodeError as e:
        return None, "derive.py 输出不是 JSON：%s" % e


# ------------------------------------------------------------------ 统一入口

def _from_derive(card, shape):
    roof, why = run_derive(card, shape)
    if roof is None:
        return None, why
    t_floor = _num(roof.get("ceiling_us"))
    bound = roof.get("cost_bound") or roof.get("bound")
    if t_floor is None:
        return None, "derive 输出里没有 ceiling_us（模型没给地板）"
    f = Floor(source="derive", bound=str(bound or "") or None,
              basis=str(roof.get("ceiling_basis") or "derive 模型地板"),
              geometry=dict(roof.get("geometry") or {}),
              inputs={"shape": shape, "model": roof.get("model"),
                      "design": dict(roof.get("design") or {})},
              detail=roof)
    f.gate = None                      # 由 _fill_accounting 按 remaining 判
    return _fill_accounting(f, t_floor=t_floor), None


def _from_roofline(card):
    d = _CARD.load_roofline(card)
    if not d:
        return None, "没有 roofline.json（derive.py 的离线产物）"
    t_floor = _num(d.get("t_floor_us")) or _num(d.get("ceiling_us"))
    remaining = _num(d.get("remaining"))
    t_actual = _num(d.get("t_actual_us"))
    if t_floor is None and remaining is None:
        return None, ("roofline.json 里既没有 t_floor_us/ceiling_us 也没有 "
                      "remaining —— 它只是结论（bound=%r），没有可算的地板"
                      % d.get("bound"))
    if remaining is None and t_floor and t_actual:
        remaining = t_actual / t_floor - 1.0
    f = Floor(source="roofline", bound=str(d.get("bound") or "") or None,
              basis=str(d.get("evidence") or d.get("bound_reason") or
                        "roofline.json 的 bound（derive.py 产出）"),
              geometry=dict(d.get("geometry") or {}), detail=d)
    f.remaining = remaining
    f.gate = gate_of(remaining)[0] if remaining is not None else None
    return _fill_accounting(f, t_floor=t_floor, t_actual=t_actual), None


def resolve_atlas(card_id):
    """只走 ③（atlas 卡 floor_inputs），返回 (Floor|None, 原因)。

    为什么需要这个入口：`scripts/floor.py` 的 CLI 只拿到 atlas **卡 id**
    （`--card gated-add`），拿不到 skill 目录，所以没法走完整的 `resolve()`。
    没有它，CLI 就只好自己再调一遍 analyze —— 那就又变成两份实现了。
    """
    try:
        spec, why = atlas_spec(card_id)
    except SystemExit as e:                       # 没有这张 atlas 卡
        return None, str(e)
    if spec is None:
        return None, why
    try:
        r = analyze(spec, load_ladder())
    except SystemExit as e:                       # 缺 bw-ladder / 峰值常数
        return None, str(e)
    if "error" in r:
        return None, r["error"]
    f = Floor(source="atlas", bound=r.get("binding"),
              basis=r.get("basis") or "", geometry={},
              inputs=spec, detail=r)
    f.gate = r.get("gate")
    return _fill_accounting(f, t_floor=r.get("t_floor_us"),
                            t_actual=r.get("t_actual_us"),
                            efficiency=r.get("utilization"),
                            remaining=r.get("remaining")), None


def _from_atlas(card):
    return resolve_atlas(_CARD.atlas_id(card))


def resolve(card, shape=None, sources=SOURCES):
    """按 `sources` 的顺序取第一个算得出来的地板。返回 Floor（永远有对象）。

    取不到时 `f.ok is False`，`f.missing` 里逐条写着「哪个来源、为什么不行」——
    这是排查「为什么这张卡算不出地板」的唯一入口，别把它吞掉。

    部分成功（有地板、没实测耗时）**不算失败**：`t_actual_us` 为 None，
    remaining 也为 None。把「有上限没实测」和「连上限都没有」混成一个
    `None` 是这类工具最常见的撒谎方式。
    """
    card = Path(card)
    missing = []
    for src in sources:
        if src == "derive":
            f, why = _from_derive(card, shape)
        elif src == "roofline":
            f, why = _from_roofline(card)
        elif src == "atlas":
            f, why = _from_atlas(card)
        else:
            f, why = None, "未知来源 %r（可选 %s）" % (src, list(SOURCES))
        if f is not None:
            return f
        missing.append("%s: %s" % (src, why))
    return Floor(missing=missing, basis="；".join(missing))

# ------------------------------------------------------------------ 自检

def selftest():
    """不依赖真实卡片与 GPU 的口径自检（`floor.py --selftest` 跑的就是它）。

    这些断言是 floor 判据的**回归测试**：它们守的不是"代码能跑"，
    而是「两个地板取慢的」「launch 是加性项」「归约要打折」这些已经被
    真实事故验证过的口径。搬到 engine/ 时一个字没改 —— 改了就说明我在
    顺手调判据，那不属于重构该干的事。
    """
    fake = {"layers": {
        "2_rw_1to1": {"achievable": 1200, "confidence": "high"},
        "2_read_only": {"achievable": 2000, "confidence": "high"},
        "2_rw_2to1": {"achievable": 1730, "confidence": "high"},
        "3_cross_row": {"achievable": None, "reason": "未标定"},
        "3_scatter_gather": {"achievable": None, "reason": "未标定"},
    }}
    # 刚好在地板：268.4MB / 1200 GB/s = 223.7 µs
    r = analyze({"bytes_mb": 268.4, "pattern": "rw_1to1", "t_actual_us": 223.7}, fake)
    assert abs(r["remaining"]) < 0.001, r
    assert r["gate"] == "at_floor", r
    print("[ok] 贴地板 → at_floor（remaining=%.3f）" % r["remaining"])

    r = analyze({"bytes_mb": 268.4, "pattern": "rw_1to1", "t_actual_us": 223.7 * 1.15}, fake)
    assert r["gate"] == "structural", r
    print("[ok] 15%% 空间 → 结构变换（remaining=%.3f）" % r["remaining"])

    r = analyze({"bytes_mb": 268.4, "pattern": "rw_1to1", "t_actual_us": 223.7 * 1.40}, fake)
    assert r["gate"] == "headroom", r
    assert "先怀疑" in r["advice"], r
    print("[ok] 40%% 空间 → 先怀疑基线/天花板（remaining=%.3f）" % r["remaining"])

    r = analyze({"bytes_mb": 268.4, "pattern": "rw_1to1", "t_actual_us": 200.0}, fake)
    assert r["gate"] == "below_floor", r
    print("[ok] 快过地板 → below_floor（remaining=%.3f）" % r["remaining"])

    r = analyze({"bytes_mb": 268.4, "pattern": "scatter_gather", "t_actual_us": 100.0}, fake)
    assert "error" in r, r
    print("[ok] 未标定模式 → 拒绝出数：%s" % r["error"])

    # 算力绑定：GEMM 4096³ 理想 2*4096^3/279e12 = 492 µs；
    # 只用带宽地板会得到 58 µs（假空间 8x），binding 必须选算力那条。
    r = analyze({"bytes_mb": 100.7, "pattern": "rw_2to1", "flops": 2 * 4096 ** 3,
                 "peak_tf": 279.0, "t_actual_us": 505.0}, fake)
    assert r["binding"] == "compute", r
    assert r["gate"] == "at_floor", r
    print("[ok] 算力绑定 → binding=compute（remaining=%.3f，带宽腿松 %.1fx）"
          % (r["remaining"], r["t_floor_us"] / 58.2))

    # 反过来：带宽绑定的算子不能因为算了 flops 就变成算力绑定
    r = analyze({"bytes_mb": 268.4, "pattern": "rw_1to1",
                 "flops": 1.0e9, "peak_tf": 279.0, "t_actual_us": 242.2}, fake)
    assert r["binding"] == "memory", r
    print("[ok] 带宽绑定 → binding=memory（remaining=%.3f）" % r["remaining"])

    # launch 地板是加性项：小 kernel 的地板是「搬运 + 43 µs」，不是只算搬运。
    # 不加它，MB 级数据会算出几百个百分点的假空间。
    r_fl = {"layers": dict(fake["layers"])}
    r_fl["layers"]["3_row_reduction"] = {"penalty_ratio": 0.79}
    no_l = analyze({"bytes_mb": 0.5, "pattern": "rw_1to1", "t_actual_us": 50.0}, r_fl)
    w_l = analyze({"bytes_mb": 0.5, "pattern": "rw_1to1", "t_actual_us": 50.0,
                   "launch_us": 43.0}, r_fl)
    assert no_l["remaining"] > 50, "0.5MB/1200GB/s 的地板极小，remaining 必然爆表"
    assert w_l["remaining"] < 1.0, "带 launch 后 remaining 应落回可用区间"
    print("[ok] launch 加性项 → remaining %.0f%% → %.1f%%（不加会给出假空间）"
          % (no_l["remaining"] * 100, w_l["remaining"] * 100))

    # topk 那类「已低于空 kernel 地板」必须老实判 below_floor，不许掩盖
    r = analyze({"bytes_mb": 0.05, "pattern": "rw_1to1", "t_actual_us": 34.2,
                 "launch_us": 43.0}, r_fl)
    assert r["gate"] == "below_floor", r
    print("[ok] 低于 launch 地板 → 老实报 below_floor（topk 34.2us < 43us）")

    # 归约折扣必须真的放宽地板，否则成熟实现会算出 +50% 假空间
    a_no = analyze({"bytes_mb": 268.4, "pattern": "rw_1to1", "t_actual_us": 242.2}, r_fl)
    a_red = analyze({"bytes_mb": 268.4, "pattern": "rw_1to1", "t_actual_us": 242.2,
                     "is_reduction": True}, r_fl)
    assert a_red["t_floor_us"] > a_no["t_floor_us"], "归约折扣应把地板放宽"
    assert a_red["remaining"] < a_no["remaining"], "归约折扣应让 remaining 变小"
    print("[ok] 归约折扣 0.79 → 地板 %.1f → %.1f us，remaining %+.1f%% → %+.1f%%"
          % (a_no["t_floor_us"], a_red["t_floor_us"],
             a_no["remaining"] * 100, a_red["remaining"] * 100))
    print("\nselftest PASS")
    return 0
