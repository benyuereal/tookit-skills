#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""efficiency — MFU / MBU 利用率计算器（分母带来源，放置条件必填）。

    MFU = achieved_flops / peak_flops     # Model FLOPs Utilization
    MBU = achieved_bw    / peak_bw        # Model Bandwidth Utilization

这个文件存在的唯一理由是「分母」和「测量条件」在别处被漏掉过很多次。
所以它不是单纯的除法：三条硬规则写死在代码里，违反就返回 {"error": ...}，
不猜、不取默认值、不把 >1 的结果截成 1.0。

硬规则
------
1. 分母必须带来源三件套 `{id, value, date}`，且 caliber 必须是 achievable：
     nominal（1100 标称带宽 / 560 标称算力）是规格上限，不可复跑，不当分母；
     impl_measured（49.8 TF 那条 aiter opus）只代表一条实现，不当分母。
   缺 id / 缺值 / 缺日期 / 日期格式不合法 → 拒绝计算。

2. 放置条件缺失时拒绝计算（efficiency_from_record）。
   算 MFU/MBU 必须知道这次测量跑在什么条件下：
   grid / ctas_per_cu / vgpr / lds_per_cta。缺任何一个、或为 null（未知）、
   或 vgpr==0（错误值，不是缺省）→ 拒绝。不猜一个默认值。

3. 精度必须匹配：bf16 上限 279 TF、fp8 上限 579.7 TF 是两个数。
   用 bf16 的分母算 fp8 的达成率会得 178% —— 本模块对 MFU/MBU > 1.0
   一律报「分母错」，不当成正常值。

用法
----
    python3 -I scripts/efficiency.py --flops 260.5 --precision bf16
    python3 -I scripts/efficiency.py --bw 1108 --pattern rw_1to1
    python3 -I scripts/efficiency.py --record <file.json> --index 0
    python3 -I scripts/efficiency.py --selftest

退出码：0 = 算出结果；1 = 拒绝计算（结果里有 error）；2 = 用法/文件错误。
"""
import argparse
import datetime
import json
import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONSTANTS = REPO / "Plugins/ops/skills/op-roofline/constants.yaml"
DEFAULT_LADDER = REPO / "Plugins/ops/op-atlas/bw-ladder.yaml"
CARDS_DIR = REPO / "Plugins/ops/op-atlas/cards"

# 算力峰值：精度 → constants.yaml 里的键路径。bf16 与 fp8 是两个独立的数，
# 绝不共用（279 vs 579.7；混用会差 2 倍）。
PRECISION_ALIASES = {
    "bf16": "bf16",
    "bfloat16": "bf16",
    "fp8": "fp8",
    "float8": "fp8",
}
# 每个精度允许的分母 id 前缀；算完后复核一遍，防止自定义 constants 把两条线接反。
PRECISION_ID_PREFIX = {"bf16": "compute.bf16", "fp8": "compute.fp8"}

# 访存模式 → bw-ladder 挡位。1_nominal 单独列出来是为了能明确拒绝它，
# 而不是「未知模式」那种含糊报错。
PATTERN_LAYERS = {
    "r_only": "2_read_only",
    "read_only": "2_read_only",
    "read": "2_read_only",
    "w_only": "2_write_only",
    "write_only": "2_write_only",
    "write": "2_write_only",
    "rw_1to1": "2_rw_1to1",
    "rw_2to1": "2_rw_2to1",
    "1_nominal": "1_nominal",
    "nominal": "1_nominal",
}
PATTERN_NAMES = ("r_only", "w_only", "rw_1to1", "rw_2to1")
# 产品标称档：文件里自己写着「只作参考，永不当分母」。
NOMINAL_LAYERS = ("1_nominal",)

PLACEMENT_REQUIRED = ("vgpr", "lds_per_cta", "ctas_per_cu", "grid")

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_MISSING = object()


class EfficiencyError(ValueError):
    """所有「拒绝计算」都走这里，最终转成 {"error": "..."}。"""


# --------------------------------------------------------------- 基础校验

def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _positive_number(v, what):
    if not _is_number(v):
        raise EfficiencyError("%s 必须是数字，实际 %r（%s）"
                              % (what, v, type(v).__name__))
    v = float(v)
    if not math.isfinite(v):
        raise EfficiencyError("%s=%r 不是有限数" % (what, v))
    if v <= 0:
        raise EfficiencyError("%s=%r 必须 > 0（0/负数不是有效测量值）" % (what, v))
    return v


def _iso_date(val, where):
    """返回 YYYY-MM-DD 字符串；日期缺失或格式不合法都抛错。

    YAML 会把 `verified_on: 2026-11-23` 解析成 datetime.date，
    也可能有人写成字符串或带时间；只接受这两种能归一成 ISO 日期的形态。
    """
    if isinstance(val, datetime.datetime):
        return val.date().isoformat()
    if isinstance(val, datetime.date):
        return val.isoformat()
    if isinstance(val, str):
        s = val.strip()
        if _DATE_RE.match(s):
            try:
                datetime.date.fromisoformat(s)
            except ValueError:
                raise EfficiencyError("%s 的日期 %r 不是合法日历日期" % (where, val))
            return s
    raise EfficiencyError(
        "%s 缺少合法来源日期：需要 YYYY-MM-DD（实际 %r）。"
        "没有来源日期的分母一律拒绝 —— 本仓因分母错吃过五次亏。" % (where, val))


def _make_denominator(den_id, value, date, caliber, unit, source):
    """构造分母三件套。任何一项不达标就抛错，绝不放行。"""
    if not isinstance(den_id, str) or not den_id.strip():
        raise EfficiencyError("分母缺少 id（来源键名），拒绝计算")
    den_id = den_id.strip()
    if caliber != "achievable":
        raise EfficiencyError(
            "%s 的 caliber=%r：达成率分母只认 achievable。"
            "nominal 是规格上限（不可复跑），impl_measured 是某一条实现的值 —— "
            "两者都不许当分母。详见 constants.yaml 三层口径。" % (den_id, caliber))
    return {
        "id": den_id,
        "value": _positive_number(value, "分母 %s.value" % den_id),
        "date": _iso_date(date, "分母 %s.date" % den_id),
        "caliber": "achievable",
        "unit": unit,
        "source": str(source or ""),
    }


def _load_yaml(path, what):
    try:
        import yaml
    except ImportError:                                  # pragma: no cover
        raise EfficiencyError("需要 PyYAML 读取 %s" % what)
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        raise EfficiencyError("读不到 %s：%s" % (p, e))
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise EfficiencyError("%s 不是合法 YAML：%s" % (p, e))
    if not isinstance(doc, dict):
        raise EfficiencyError("%s 顶层必须是 mapping" % p)
    return doc


# --------------------------------------------------------------- 分母来源

def resolve_peak_denominator(precision, constants_path=None):
    """精度 → achievable 算力峰值三件套。

    bf16: compute.bf16_gemm_peak_tf.value         = 279.0 TF
    fp8 : compute.fp8_peak_layers.achievable.value = 579.7 TF
    两者不共用：拿 bf16 的 279 算 fp8 会得出 178%。
    """
    canonical = PRECISION_ALIASES.get(str(precision).strip().lower())
    if canonical is None:
        raise EfficiencyError(
            "不支持的精度 %r：只支持 bf16 / fp8。"
            "每种精度有自己的 achievable 峰值，不许互相顶替。" % (precision,))
    path = Path(constants_path) if constants_path else DEFAULT_CONSTANTS
    doc = _load_yaml(path, "算力常数 constants.yaml")
    comp = doc.get("compute")
    if not isinstance(comp, dict):
        raise EfficiencyError("%s 缺 compute 段" % path)

    if canonical == "bf16":
        node = comp.get("bf16_gemm_peak_tf")
        if not isinstance(node, dict):
            raise EfficiencyError("%s 缺 compute.bf16_gemm_peak_tf" % path)
        return _make_denominator(
            "compute.bf16_gemm_peak_tf",
            node.get("value"),
            node.get("verified_on"),
            node.get("caliber"),
            node.get("unit") or "TFLOPS",
            node.get("source"))

    outer = comp.get("fp8_peak_layers")
    if not isinstance(outer, dict):
        raise EfficiencyError("%s 缺 compute.fp8_peak_layers" % path)
    inner = outer.get("achievable")
    if isinstance(inner, dict):
        value = inner.get("value", outer.get("achievable_tf"))
        date = inner.get("verified_on", outer.get("verified_on"))
        caliber = inner.get("caliber", "achievable")
        unit = inner.get("unit") or "TFLOPS"
        source = inner.get("source")
    elif outer.get("achievable_tf") is not None:
        # 只写了 achievable_tf 的紧凑写法：日期/口径取外层的。
        value = outer.get("achievable_tf")
        date = outer.get("verified_on")
        caliber = outer.get("caliber", "achievable")
        unit = outer.get("unit") or "TFLOPS"
        source = outer.get("source")
    else:
        raise EfficiencyError(
            "%s 的 fp8_peak_layers 没有 achievable 值 —— "
            "nominal(560)/isa_ceiling(686.5)/impl_measured(49.8) 都不是达成率分母"
            % path)
    return _make_denominator(
        "compute.fp8_peak_layers.achievable_tf", value, date, caliber, unit, source)


def resolve_bw_denominator(pattern, ladder_path=None):
    """访存模式 → achievable 带宽三件套（四档，不取平均）。

    r_only   1965（2_read_only）
    w_only   1692（2_write_only）
    rw_1to1  1680（2_rw_1to1）
    rw_2to1  1730（2_rw_2to1）
    1_nominal 1100 是产品标称，永不当分母。
    """
    key = str(pattern).strip().lower()
    layer = PATTERN_LAYERS.get(key)
    if layer is None:
        raise EfficiencyError(
            "未知访存模式 %r；可选 %s（读写比不同必须换档，"
            "不可拿纯读 1965 当所有算子的分母）" % (pattern, ", ".join(PATTERN_NAMES)))
    if layer in NOMINAL_LAYERS:
        raise EfficiencyError(
            "访存模式 %r 指向 1_nominal（产品标称 1100 GB/s）："
            "它不是 achievable，永不当分母。实测四档见 bw-ladder.yaml，"
            "请按读写比选 %s。" % (pattern, ", ".join(PATTERN_NAMES)))

    path = Path(ladder_path) if ladder_path else DEFAULT_LADDER
    doc = _load_yaml(path, "带宽梯子 bw-ladder.yaml")
    layers = doc.get("layers")
    if not isinstance(layers, dict):
        raise EfficiencyError("%s 缺 layers 段" % path)
    node = layers.get(layer)
    if not isinstance(node, dict):
        raise EfficiencyError("%s 缺 layers.%s" % (path, layer))
    if node.get("confidence") == "n/a":
        raise EfficiencyError(
            "%s 的 %s confidence=n/a（标称/不可复跑），不当分母" % (path, layer))
    value = node.get("achievable")
    if value is None:
        raise EfficiencyError(
            "%s 的 %s 没有 achievable 值（status=%r, reason=%r）："
            "未标定的档位不当分母" % (path, layer, node.get("status"),
                                     node.get("reason")))
    return _make_denominator(
        "bw-ladder.layers.%s.achievable" % layer,
        value,
        doc.get("date"),
        node.get("caliber", "achievable"),
        "GB/s",
        "%s (confidence=%s, impl=%s)" % (path.name, node.get("confidence"),
                                         node.get("impl")))


# --------------------------------------------------------------- MFU / MBU

def compute_mfu(achieved_flops, precision, constants_path=None):
    """achieved_flops 单位 TFLOPS。返回 {"mfu", "numerator", "denominator"} 或 {"error"}。"""
    try:
        canonical = PRECISION_ALIASES.get(str(precision).strip().lower())
        if canonical is None:
            raise EfficiencyError(
                "不支持的精度 %r：只支持 bf16 / fp8" % (precision,))
        num = _positive_number(achieved_flops, "achieved_flops(TFLOPS)")
        den = resolve_peak_denominator(canonical, constants_path)
        # 复核：分母 id 必须属于该精度，防止自定义 constants 接反两条线。
        prefix = PRECISION_ID_PREFIX[canonical]
        if not den["id"].startswith(prefix):
            raise EfficiencyError(
                "分母 %s 不属于精度 %s（期望前缀 %s）：精度与分母不匹配，拒绝计算"
                % (den["id"], canonical, prefix))
        mfu = num / den["value"]
        if mfu > 1.0 + 1e-9:
            raise EfficiencyError(
                "MFU=%.4f > 1.0（%.6g / %.6g TFLOPS）：分母错，不是突破。"
                "先查精度是否配错 —— bf16 上限 279 TF、fp8 上限 579.7 TF 是两个数，"
                "拿 bf16 的分母算 fp8 会得出 178%%（%.6g/279=%.3f）；"
                "再查 achieved_flops 的单位（本接口是 TFLOPS）。分母来源：%s"
                % (mfu, num, den["value"], num, num / 279.0, den["id"]))
        return {
            "mfu": round(mfu, 6),
            "numerator": {
                "achieved_flops_tflops": num,
                "precision": canonical,
            },
            "denominator": den,
        }
    except EfficiencyError as e:
        return {"error": str(e)}


def compute_mbu(achieved_bw, pattern, ladder_path=None):
    """achieved_bw 单位 GB/s。返回 {"mbu", "numerator", "denominator"} 或 {"error"}。"""
    try:
        num = _positive_number(achieved_bw, "achieved_bw(GB/s)")
        den = resolve_bw_denominator(pattern, ladder_path)
        mbu = num / den["value"]
        if mbu > 1.0 + 1e-9:
            raise EfficiencyError(
                "MBU=%.4f > 1.0（%.6g / %.6g GB/s）：分母错，不是突破。"
                "先查是否用了 1_nominal(1100 标称) 或错的读写档（四档 "
                "1965/1692/1680/1730 不可混用），再查 achieved_bw 的字节口径。"
                "分母来源：%s" % (mbu, num, den["value"], den["id"]))
        return {
            "mbu": round(mbu, 6),
            "numerator": {
                "achieved_bw_gbs": num,
                "pattern": str(pattern).strip().lower(),
            },
            "denominator": den,
        }
    except EfficiencyError as e:
        return {"error": str(e)}


# --------------------------------------------------------------- record 入口

def _grid_ctas(val):
    """grid → 总 CTA 数；非法返回 None。整数就是总数，列表取积。"""
    if isinstance(val, int) and not isinstance(val, bool):
        return val if val >= 1 else None
    if isinstance(val, list):
        if not val:
            return None
        if any(not (isinstance(x, int) and not isinstance(x, bool) and x >= 1)
               for x in val):
            return None
        return math.prod(val)
    return None


def _require_placement(record):
    """取 placement 并逐字段校验；缺/未知/错误值一律抛错。"""
    if not isinstance(record, dict):
        raise EfficiencyError("record 必须是 JSON 对象，实际 %s"
                              % type(record).__name__)
    if "placement" not in record:
        raise EfficiencyError(
            "record 缺 placement：算 MFU/MBU 必须知道测量条件（%s），"
            "不许用默认值" % "/".join(PLACEMENT_REQUIRED))
    plc = record["placement"]
    if not isinstance(plc, dict):
        raise EfficiencyError("record.placement 必须是对象，实际 %s"
                              % type(plc).__name__)
    unk = record.get("placement_unknown")
    if not isinstance(unk, dict):
        unk = {}

    for f in PLACEMENT_REQUIRED:
        if f not in plc:
            raise EfficiencyError(
                "record.placement 缺 %s：放置条件不全拒绝计算"
                "（未知请写 null + placement_unknown 原因，但仍不能算）" % f)
        v = plc[f]
        if v is None:
            reason = unk.get(f)
            raise EfficiencyError(
                "record.placement.%s 为 null（原因：%s）：放置条件未知，"
                "拒绝计算。MFU/MBU 只在已知放置下有意义，不许推算一个。"
                % (f, reason.strip() if isinstance(reason, str) and reason.strip()
                   else "未给原因"))

    if _grid_ctas(plc["grid"]) is None:
        raise EfficiencyError(
            "record.placement.grid=%r 非法：必须是 ≥1 的整数（总 CTA 数）"
            "或整数列表（各维 CTA 数）" % (plc["grid"],))
    for f in ("ctas_per_cu", "vgpr", "lds_per_cta"):
        v = plc[f]
        if not (isinstance(v, int) and not isinstance(v, bool)):
            raise EfficiencyError("record.placement.%s=%r 必须是整数" % (f, v))
    if plc["ctas_per_cu"] < 1:
        raise EfficiencyError(
            "record.placement.ctas_per_cu=%r 必须 ≥1；0 意味着 kernel 不驻留"
            % (plc["ctas_per_cu"],))
    if plc["vgpr"] == 0:
        raise EfficiencyError(
            "record.placement.vgpr=0 是错误值，不是「没填」："
            "VGPR*threads=0 会让占用率公式静默变成无限 CTA。"
            "实测值取自 inspect_so.py 的 .vgpr_count；未知写 null（但仍不能算）")
    if plc["vgpr"] < 0:
        raise EfficiencyError("record.placement.vgpr=%r 不能为负" % (plc["vgpr"],))
    if plc["lds_per_cta"] < 0:
        raise EfficiencyError(
            "record.placement.lds_per_cta=%r 不能为负（0 合法：不用 LDS）"
            % (plc["lds_per_cta"],))
    if "waves_per_cu" in plc and plc["waves_per_cu"] is not None:
        w = plc["waves_per_cu"]
        if not (isinstance(w, int) and not isinstance(w, bool)) or w < 1:
            raise EfficiencyError(
                "record.placement.waves_per_cu=%r 非法（必须是 ≥1 的整数）" % (w,))
    return plc


def _first_number(record, keys):
    for k in keys:
        v = record.get(k)
        if _is_number(v):
            return float(v)
    return None


def _card_meta(card):
    """card 可以是 dict / 路径 / card id；只取显式字段，不解析散文。"""
    if card is None:
        return {}
    if isinstance(card, dict):
        doc = card
    else:
        p = Path(str(card))
        if not p.is_file():
            p = CARDS_DIR / (str(card) + ".yaml")
        if not p.is_file():
            raise EfficiencyError("找不到 card %r（%s）" % (card, CARDS_DIR))
        doc = _load_yaml(p, "card %s" % p)
    meta = {}
    for k in ("precision", "pattern", "achieved_bw", "achieved_flops"):
        if doc.get(k) is not None:
            meta[k] = doc[k]
    return meta


def efficiency_from_record(record, card=None, constants_path=None,
                           ladder_path=None):
    """从一条 results/perf record 直接算 MFU 和/或 MBU；缺 placement 就拒。

    分子来源（按优先级）：
      achieved_flops / achieved_tflops / achieved_tf / tflops   [TFLOPS]
      achieved_bw / achieved_gbs / bw_gbs                       [GB/s]
    若只有 median_us + flops（总 FLOPs）或 bytes（总字节）→ 现算速率。
    精度/模式取 record 的 precision/pattern，缺时用 card 的显式字段补。
    placement **只从 record 取**，card 不能替它作证。
    """
    try:
        plc = _require_placement(record)
        meta = _card_meta(card)

        precision = record.get("precision") or meta.get("precision")
        pattern = record.get("pattern") or meta.get("pattern")
        flops = _first_number(record, ("achieved_flops", "achieved_tflops",
                                       "achieved_tf", "tflops"))
        bw = _first_number(record, ("achieved_bw", "achieved_gbs", "bw_gbs"))
        if bw is None and _is_number(meta.get("achieved_bw")):
            bw = float(meta["achieved_bw"])
        if flops is None and _is_number(meta.get("achieved_flops")):
            flops = float(meta["achieved_flops"])

        median_us = _first_number(record, ("median_us", "time_us",
                                           "latency_us", "us"))
        if flops is None and median_us and _is_number(record.get("flops")):
            flops = float(record["flops"]) / (median_us * 1e-6) / 1e12
        if bw is None and median_us and _is_number(record.get("bytes")):
            bw = float(record["bytes"]) / (median_us * 1e-6) / 1e9

        if flops is None and bw is None:
            raise EfficiencyError(
                "record 里没有可用的分子：需要 achieved_flops(TFLOPS) 或 "
                "achieved_bw(GB/s)，或 median_us + flops/bytes 现算")

        out = {"placement": {k: plc[k] for k in plc}}
        if flops is not None:
            if not precision:
                raise EfficiencyError(
                    "record 有算力但没有 precision：不知道用 279(bf16) 还是 "
                    "579.7(fp8) 作分母，拒绝计算")
            r = compute_mfu(flops, precision, constants_path)
            if "error" in r:
                raise EfficiencyError("MFU: " + r["error"])
            out.update(r)
        if bw is not None:
            if not pattern:
                raise EfficiencyError(
                    "record 有带宽但没有 pattern：不知道用哪一档 "
                    "（1965/1692/1680/1730），拒绝计算")
            r = compute_mbu(bw, pattern, ladder_path)
            if "error" in r:
                raise EfficiencyError("MBU: " + r["error"])
            if "mfu" in out:
                out["mbu"] = r["mbu"]
                out["mbu_numerator"] = r["numerator"]
                out["mbu_denominator"] = r["denominator"]
            else:
                out["mbu"] = r["mbu"]
                out["numerator"] = r["numerator"]
                out["denominator"] = r["denominator"]
        return out
    except EfficiencyError as e:
        return {"error": str(e)}


# --------------------------------------------------------------- CLI

def _cli_flops_to_tflops(value, unit="tflops"):
    """--flops 的单位必须**显式给**，不许按大小猜。

    第一版按「≥1e6 视为 FLOP/s」自动换算——那是在替用户猜单位。
    `--flops 1.37e11` 本意是 137 GFLOP 总量（应除以耗时得 TFLOPS），
    却被当成速率换算成 0.137 TFLOPS，MFU 变成 0.0005。
    静默猜测单位正是本仓反复栽的那类错（cf. 1724 当分母、49.8 当上限）。
    """
    u = str(unit).lower()
    if u not in ("tflops", "flops_per_s"):
        raise ValueError("--flops-unit 只能是 tflops 或 flops_per_s，实得 %r" % unit)
    if u == "flops_per_s":
        return value / 1e12, "flops_per_s"
    return value, "tflops"


def _print(obj):
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def _has_error(obj):
    if not isinstance(obj, dict):
        return True
    if "error" in obj:
        return True
    return False


def selftest():
    """不碰 GPU、不写文件。真常数验算 + 六条反向用例。"""
    checks = []

    r = compute_mfu(260.5, "bf16")
    assert "error" not in r, r
    assert abs(r["mfu"] - 260.5 / 279.0) < 1e-6, r
    assert r["denominator"]["id"] == "compute.bf16_gemm_peak_tf", r
    assert r["denominator"]["value"] == 279.0, r
    assert r["denominator"]["date"] == "2026-11-23", r
    checks.append("[ok] bf16：260.5 TF / 279 = %.6f，分母 id/value/date 齐全"
                  % r["mfu"])

    r = compute_mfu(498.5, "fp8")
    assert "error" not in r, r
    assert abs(r["mfu"] - 498.5 / 579.7) < 1e-6, r
    assert r["denominator"]["id"] == "compute.fp8_peak_layers.achievable_tf", r
    assert r["denominator"]["value"] == 579.7, r
    checks.append("[ok] fp8：498.5 TF / 579.7 = %.6f（用的是 fp8 自己的分母）"
                  % r["mfu"])

    r = compute_mfu(498.5, "bf16")
    assert "error" in r and "1.7867" in r["error"], r
    checks.append("[ok] 拿 bf16 的 279 算 fp8(498.5 TF) → MFU 1.79 > 1，"
                  "报「分母错/精度配错」而不是静默给 178%")

    r = compute_mbu(1108, "rw_1to1")
    assert "error" not in r, r
    assert abs(r["mbu"] - 1108 / 1680.0) < 1e-6, r
    assert r["denominator"]["value"] == 1680.0, r
    assert r["denominator"]["date"] == "2026-11-23", r
    checks.append("[ok] rw_1to1：1108 / 1680 = %.6f" % r["mbu"])

    r = compute_mbu(1108, "1_nominal")
    assert "error" in r and "标称" in r["error"], r
    checks.append("[ok] 用 1_nominal(1100 标称) 当带宽分母 → 拒绝")

    r = compute_mbu(1108, "rw_1to1", ladder_path="/nonexistent.yaml")
    assert "error" in r, r
    checks.append("[ok] 带宽梯子缺失 → 拒绝（不猜档位）")

    base = {"placement": {"grid": [128, 1], "ctas_per_cu": 4, "vgpr": 96,
                          "lds_per_cta": 16384, "waves_per_cu": 16}}
    rec = dict(base, achieved_bw=1108, pattern="rw_1to1")
    r = efficiency_from_record(rec)
    assert "error" not in r and abs(r["mbu"] - 0.659524) < 1e-6, r
    checks.append("[ok] 完整 record → MBU=%.6f（placement 一并回显）" % r["mbu"])

    bad = {"placement": {"grid": 128, "ctas_per_cu": 4,
                         "lds_per_cta": 16384}, "achieved_bw": 1108,
           "pattern": "rw_1to1"}
    r = efficiency_from_record(bad)
    assert "error" in r and "vgpr" in r["error"], r
    checks.append("[ok] placement 缺 vgpr → 拒绝计算")

    bad = dict(base, achieved_bw=1108, pattern="rw_1to1")
    bad["placement"] = dict(base["placement"], vgpr=0)
    r = efficiency_from_record(bad)
    assert "error" in r and "vgpr=0" in r["error"], r
    checks.append("[ok] vgpr=0 → 拒绝（错误值，不是缺省）")

    print("\n".join(checks))
    print("\nselftest PASS（%d 项）" % len(checks))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="MFU/MBU 计算器：分母必须带来源、放置条件必须实测")
    ap.add_argument("--flops-unit", choices=("tflops", "flops_per_s"),
                    default="tflops", help="--flops 的单位，必须显式给（默认 tflops，不按大小猜）")
    ap.add_argument("--flops", type=float,
                    help="达成的算力；TFLOPS（≥1e6 按 FLOP/s 自动换算）")
    ap.add_argument("--precision", help="算力精度：bf16 / fp8（决定分母 279 / 579.7）")
    ap.add_argument("--bw", type=float, help="达成的带宽，GB/s")
    ap.add_argument("--pattern", help="访存模式：%s" % ", ".join(PATTERN_NAMES))
    ap.add_argument("--record", metavar="FILE.json",
                    help="results/perf record 文件（按 placement 校验后计算）")
    ap.add_argument("--index", type=int, default=0, help="record 文件里的下标（默认 0）")
    ap.add_argument("--card", help="card id / 路径 / JSON 对象文件，补 precision/pattern")
    ap.add_argument("--constants", help="覆盖 constants.yaml 路径")
    ap.add_argument("--ladder", help="覆盖 bw-ladder.yaml 路径")
    ap.add_argument("--selftest", action="store_true", help="无需 GPU/文件的自检")
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()

    if a.record:
        p = Path(a.record)
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except OSError as e:
            print("ERROR: 读不到 %s：%s" % (p, e), file=sys.stderr)
            return 2
        except ValueError as e:
            print("ERROR: %s 不是合法 JSON：%s" % (p, e), file=sys.stderr)
            return 2
        if isinstance(doc, dict) and isinstance(doc.get("records"), list):
            recs = doc["records"]
            if not (0 <= a.index < len(recs)):
                print("ERROR: --index %d 越界（共 %d 条 record）"
                      % (a.index, len(recs)), file=sys.stderr)
                return 2
            rec = recs[a.index]
            card = a.card
            if card is None and isinstance(doc.get("card"), str):
                cid = doc["card"]
                if (CARDS_DIR / (cid + ".yaml")).is_file():
                    card = cid
        else:
            if a.index != 0:
                print("ERROR: 单条 record 只能用 --index 0", file=sys.stderr)
                return 2
            rec, card = doc, a.card
        out = efficiency_from_record(rec, card=card,
                                     constants_path=a.constants,
                                     ladder_path=a.ladder)
        _print(out)
        return 1 if _has_error(out) else 0

    out = {}
    if a.flops is not None:
        if not a.precision:
            ap.error("--flops 需要 --precision {bf16,fp8}")
        try:
            tflops, unit = _cli_flops_to_tflops(a.flops, getattr(a, "flops_unit", "tflops"))
        except ValueError as ex:
            print("error: %s" % ex, file=sys.stderr)
            return 2
        if unit != "tflops":
            print("note: --flops %.6g 按 FLOP/s 解释 → %.6g TFLOPS"
                  % (a.flops, tflops), file=sys.stderr)
        r = compute_mfu(tflops, a.precision, a.constants)
        out.update(r)
        out["numerator"] = dict(r.get("numerator") or {}, input_unit=unit)
    if a.bw is not None:
        if not a.pattern:
            ap.error("--bw 需要 --pattern {%s}" % ",".join(PATTERN_NAMES))
        r = compute_mbu(a.bw, a.pattern, a.ladder)
        out.update(r)
    if not out:
        ap.print_help()
        return 2
    _print(out)
    return 1 if _has_error(out) else 0


if __name__ == "__main__":
    sys.exit(main())
