#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""results_schema — 给 perf 实测结果强制加上「运行放置条件」。

为什么需要它
------------
`Plugins/ops/op-atlas/perf/*.json` 里的 median_us 一直只是「当时的数」：
没有记录跑在什么 grid 上、每 CU 驻留几个 CTA、VGPR/LDS 是多少。
于是结论不可迁移。两个实测反例（都在本机）：

  * hbird-gemm：同一个 kernel，136 CTA 时 396 TF，128 个均衡 CTA 时 547 TF。
    差 38% 不是 kernel 变了，是 grid 与 CU 数不整除。
  * LDS 允许 5 个 CTA，但 VGPR 只允许 3 个 —— 「LDS 允许的 CTA 数」
    不等于「实际驻留数」。只写单资源上限会把占用率算高。

所以每条 record 必须带 `placement`，五个字段：

  grid          实际 launch 的 CTA 数：整数，或各维 CTA 数的列表（取积）
  ctas_per_cu   每 CU **实际驻留** 的 CTA 数（不是单一资源允许的上限）
  vgpr          每线程 VGPR 数（正整数）
  lds_per_cta   每 CTA 的 LDS 字节数（0 合法：不用 LDS）
  waves_per_cu  每 CU 驻留的 wave 数

「未知」与「0」是两件事
-----------------------
未知字段一律写 `null`，并在 record 的 `placement_unknown` 里给原因：

    "placement": {"grid": null, ...},
    "placement_unknown": {"grid": "复现脚本没记录 launch 维度，需重跑"}

没有原因 = 与漏填无法区分，报错。**不许用 0 冒充未知**：
`vgpr: 0` 会让占用率公式 `VGPR_BUDGET/(vgpr*threads)` 静默变成「无限 CTA」。
同理 `lds_per_cta: 0` 是合法值（真的不用 LDS），而 `vgpr: 0` 不是。

用法
----
    python3 -I scripts/results_schema.py --check Plugins/ops/op-atlas/perf/*.json
    python3 -I scripts/results_schema.py --check --strict --json x.json
    python3 -I scripts/results_schema.py --fill-from-so r.json --so /tmp/k.so --threads 256
    python3 -I scripts/results_schema.py --selftest

退出码：0 = 全部通过；1 = 校验不通过（--strict 时警告也算）；2 = 用法/文件错误。
--check 只读；--fill-from-so 才写文件（--dry-run 可只看不写）。
"""
import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

PLACEMENT_FIELDS = ("grid", "ctas_per_cu", "vgpr", "lds_per_cta", "waves_per_cu")

# 本机（BW1100 / gfx938）64 CU；只用于「每 CU 能驻多少」的合理性判断。
CU_COUNT = 64
# inspect_so.py 用的每 CU LDS 预算（64 KiB），保持一致。
LDS_BUDGET = 64 * 1024
# 768-VGPR 模式下单线程 VGPR 上限；超过它只可能是字段填错。
VGPR_MAX = 768

REPO = Path(__file__).resolve().parents[1]
INSPECT_SO = REPO / "Plugins/ops/skills/op-hip-compile/inspect_so.py"

_MISSING = object()


class Problem(object):
    __slots__ = ("severity", "where", "message")

    def __init__(self, severity, where, message):
        self.severity = severity      # "error" | "warning"
        self.where = where
        self.message = message

    def as_dict(self):
        return {"severity": self.severity, "where": self.where,
                "message": self.message}

    def __str__(self):
        tag = "FAIL" if self.severity == "error" else "WARN"
        return "%s %s: %s" % (tag, self.where, self.message)


def _err(where, msg):
    return Problem("error", where, msg)


def _warn(where, msg):
    return Problem("warning", where, msg)


def _is_int(v):
    # bool 是 int 的子类；`true` 不能当 1 用。
    return isinstance(v, int) and not isinstance(v, bool)


def _grid_ctas(val):
    """grid 值 → 总 CTA 数；非法返回 None。整数直接就是总数，列表取积。"""
    if _is_int(val):
        return val if val >= 1 else None
    if isinstance(val, list):
        if not val:
            return None
        if any(not (_is_int(x) and x >= 1) for x in val):
            return None
        return math.prod(val)
    return None


def validate_record(rec, idx, num_cu=CU_COUNT, require_known=False):
    """校验单条 record 的 placement。

    返回 (problems, info)。info = {"unknown": [...], "grid_ctas": int|None,
    "ctas_per_cu": int|None, "waves_per_cu": int|None}。
    """
    probs = []
    info = {"unknown": [], "grid_ctas": None,
            "ctas_per_cu": None, "waves_per_cu": None}
    loc = "records[%d]" % idx
    if not isinstance(rec, dict):
        probs.append(_err(loc, "record 不是 JSON 对象"))
        return probs, info

    variant = rec.get("variant")
    if isinstance(variant, str) and variant.strip():
        loc += "(%s)" % variant.strip()
    else:
        probs.append(_warn(loc + ".variant", "缺少 variant 字符串，定位会变困难"))

    if "placement" not in rec:
        probs.append(_err(
            loc + ".placement",
            "缺少 placement 块；必需字段 %s（未知写 null + placement_unknown 原因）"
            % "/".join(PLACEMENT_FIELDS)))
        return probs, info

    plc = rec["placement"]
    if plc is None:
        probs.append(_err(
            loc + ".placement",
            "placement 为 null 不能表示『未知』：未知必须逐字段写 null 并给原因"))
        return probs, info
    if not isinstance(plc, dict):
        probs.append(_err(loc + ".placement", "placement 必须是对象，实际是 %s"
                          % type(plc).__name__))
        return probs, info

    unk = rec.get("placement_unknown", _MISSING)
    if unk is _MISSING:
        unk = {}
    elif not isinstance(unk, dict):
        probs.append(_err(loc + ".placement_unknown",
                          "必须是对象 {字段: 原因}，实际是 %s" % type(unk).__name__))
        unk = {}

    for k in plc:
        if k not in PLACEMENT_FIELDS:
            probs.append(_err(
                "%s.placement.%s" % (loc, k),
                "未知字段 %r；只允许 %s（拼写错误不许静默通过）"
                % (k, ", ".join(PLACEMENT_FIELDS))))
    for k in unk:
        if k not in PLACEMENT_FIELDS:
            probs.append(_err(
                "%s.placement_unknown.%s" % (loc, k),
                "未知字段 %r；只允许 %s" % (k, ", ".join(PLACEMENT_FIELDS))))

    for f in PLACEMENT_FIELDS:
        floc = "%s.placement.%s" % (loc, f)
        if f not in plc:
            probs.append(_err(
                floc,
                "缺少必需字段；未知请写 null 并在 placement_unknown.%s 给原因，"
                "不许省略、不许用 0 冒充" % f))
            continue

        val = plc[f]
        if val is None:
            reason = unk.get(f)
            if not isinstance(reason, str) or not reason.strip():
                probs.append(_err(
                    floc,
                    "值为 null 但 placement_unknown.%s 没给原因；"
                    "『未知』必须带原因，否则与漏填无法区分" % f))
            else:
                info["unknown"].append(f)
                if require_known:
                    probs.append(_err(
                        floc, "值为 null（原因：%s）；--require-known 要求实测值"
                        % reason.strip()))
            continue

        if f == "grid":
            total = _grid_ctas(val)
            if total is None:
                if isinstance(val, list) and not val:
                    probs.append(_err(floc, "grid 是空列表；至少给一维"))
                elif isinstance(val, list):
                    probs.append(_err(
                        floc, "grid 列表元素必须是 ≥1 的整数，实际 %r" % (val,)))
                else:
                    probs.append(_err(
                        floc, "grid 必须是整数（总 CTA 数）或整数列表"
                        "（各维 CTA 数），实际 %r" % (val,)))
            else:
                info["grid_ctas"] = total

        elif f == "ctas_per_cu":
            if not _is_int(val):
                probs.append(_err(floc, "必须是整数，实际 %r" % (val,)))
            elif val < 1:
                probs.append(_err(
                    floc, "ctas_per_cu=%r：每 CU 驻留的 CTA 数必须 ≥1；"
                    "0 意味着 kernel 永远不会驻留（值错或字段填错）" % (val,)))
            else:
                info["ctas_per_cu"] = val
                if val > num_cu:
                    probs.append(_warn(
                        floc,
                        "ctas_per_cu=%d > num_cu=%d：单 CU 不可能驻留超过 CU 数的 CTA；"
                        "确认这是『实际驻留』而不是总 CTA 数，"
                        "也注意 inspect_so 的 CTA/CU 公式没有 max-waves/CU 上限"
                        % (val, num_cu)))

        elif f == "vgpr":
            if not _is_int(val):
                probs.append(_err(floc, "必须是整数，实际 %r" % (val,)))
            elif val == 0:
                probs.append(_err(
                    floc,
                    "vgpr=0 是错误值，不是『没填』：VGPR*threads=0 会让占用率公式"
                    "静默变成无限 CTA。实测值取自 inspect_so.py 的 .vgpr_count；"
                    "未知请写 null + 原因"))
            elif val < 0:
                probs.append(_err(floc, "vgpr=%r 不能为负" % (val,)))
            elif val > VGPR_MAX:
                probs.append(_warn(
                    floc, "vgpr=%d 超过 768-VGPR 模式的单线程上限 %d，疑似填错字段"
                    % (val, VGPR_MAX)))

        elif f == "lds_per_cta":
            if not _is_int(val):
                probs.append(_err(floc, "必须是整数（字节），实际 %r" % (val,)))
            elif val < 0:
                probs.append(_err(floc, "lds_per_cta=%r 不能为负" % (val,)))
            elif val > LDS_BUDGET:
                probs.append(_warn(
                    floc,
                    "lds_per_cta=%d B > 每 CU LDS 预算 %d B（与 inspect_so.py 一致）；"
                    "要么填成了别的量，要么 kernel 无法 launch"
                    % (val, LDS_BUDGET)))

        elif f == "waves_per_cu":
            if not _is_int(val):
                probs.append(_err(floc, "必须是整数，实际 %r" % (val,)))
            elif val < 1:
                probs.append(_err(floc, "waves_per_cu=%r 必须 ≥1" % (val,)))
            else:
                info["waves_per_cu"] = val

    # ---- 跨字段自洽 ----
    grid = info["grid_ctas"]
    ctas = info["ctas_per_cu"]
    waves = info["waves_per_cu"]

    if grid is not None and ctas is not None and ctas > grid:
        probs.append(_warn(
            "records[%d].placement" % idx,
            "每 CU 驻留上限 ctas_per_cu=%d 大于总 CTA 数 grid=%d，实际不可能驻留这么多"
            % (ctas, grid)))

    if grid is not None and num_cu > 0:
        if grid < num_cu:
            probs.append(_warn(
                "records[%d].placement.grid" % idx,
                "总 CTA 数 %d < CU 数 %d：部分 CU 空转，占用率与实测都会被小 grid 支配"
                % (grid, num_cu)))
        elif grid % num_cu:
            probs.append(_warn(
                "records[%d].placement.grid" % idx,
                "总 CTA 数 %d 不是 CU 数 %d 的整数倍：尾部不均衡（实测 hbird-gemm "
                "136 CTA=396 TF vs 128 CTA=547 TF，同一 kernel 差 38%%）"
                % (grid, num_cu)))

    if waves is not None and ctas is not None:
        if waves < ctas:
            probs.append(_warn(
                "records[%d].placement" % idx,
                "waves_per_cu=%d < ctas_per_cu=%d：每 CTA 至少 1 个 wave，两者矛盾"
                % (waves, ctas)))
        elif waves % ctas:
            probs.append(_warn(
                "records[%d].placement" % idx,
                "waves_per_cu=%d 不是 ctas_per_cu=%d 的整数倍（每 CTA %.3f 个 wave）；"
                "同一 kernel 每 CTA 的 wave 数必须是整数，两者必有一个错"
                % (waves, ctas, float(waves) / ctas)))

    return probs, info


def validate_document(doc, path="<json>", num_cu=CU_COUNT, require_known=False):
    """校验一个 results 文档结构 + 每条 record 的 placement。

    返回 (problems, summary)。summary = {"path", "card", "records",
    "unknown_fields", "unknown_detail"}。
    """
    probs = []
    summary = {"path": str(path), "card": None, "records": 0,
               "unknown_fields": 0, "unknown_detail": []}
    if not isinstance(doc, dict):
        probs.append(_err(str(path), "顶层必须是 JSON 对象"))
        return probs, summary

    summary["card"] = doc.get("card")
    records = doc.get("records")
    if not isinstance(records, list):
        probs.append(_err(str(path), "缺少 records 数组"))
        return probs, summary
    if not records:
        probs.append(_err(str(path), "records 为空：没有实测记录的结果文件是空承诺"))
        return probs, summary

    for i, rec in enumerate(records):
        rp, info = validate_record(rec, i, num_cu=num_cu,
                                   require_known=require_known)
        probs.extend(rp)
        summary["unknown_fields"] += len(info["unknown"])
        if info["unknown"]:
            variant = rec.get("variant") if isinstance(rec, dict) else None
            summary["unknown_detail"].append({
                "record": i, "variant": variant,
                "fields": info["unknown"]})
    summary["records"] = len(records)
    return probs, summary


def _die(msg):
    """用法/文件层错误统一退出 2（与『校验不通过』的 1 区分开）。"""
    print("ERROR: " + msg, file=sys.stderr)
    raise SystemExit(2)


def load_document(path):
    """读 JSON；IO/语法错误 → exit 2，不混进校验结论。"""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        _die("读不到 %s：%s" % (p, e))
    try:
        return json.loads(text)
    except ValueError as e:
        _die("%s 不是合法 JSON：%s" % (p, e))


# ---------------------------------------------------------------- check CLI

def cmd_check(paths, num_cu=CU_COUNT, strict=False, require_known=False,
              as_json=False):
    all_probs = []
    summaries = []
    for path in paths:
        doc = load_document(path)
        probs, summary = validate_document(
            doc, path, num_cu=num_cu, require_known=require_known)
        all_probs.extend(probs)
        summaries.append(summary)

    errors = [p for p in all_probs if p.severity == "error"]
    warnings = [p for p in all_probs if p.severity == "warning"]
    failed = bool(errors) or (strict and bool(warnings))

    if as_json:
        print(json.dumps({
            "ok": not failed,
            "strict": strict,
            "problems": [p.as_dict() for p in all_probs],
            "files": summaries,
        }, indent=2, ensure_ascii=False))
    else:
        for p in all_probs:
            print(p)
        n_rec = sum(s["records"] for s in summaries)
        n_unk = sum(s["unknown_fields"] for s in summaries)
        print("\n%d 个文件 / %d 条 record：%d error, %d warning, %d 个 unknown 字段"
              % (len(summaries), n_rec, len(errors), len(warnings), n_unk))
        for s in summaries:
            if s["unknown_detail"]:
                for u in s["unknown_detail"]:
                    print("  unknown %s: records[%d](%s) -> %s"
                          % (s["path"], u["record"], u["variant"],
                             ", ".join(u["fields"])))
        if failed:
            print("结果：不通过%s" % ("（--strict：警告也算失败）" if not errors else ""))
        else:
            print("结果：通过")

    return 1 if failed else 0


# ------------------------------------------------------- fill from .so

def read_so(so_path, threads=None, lds=None, vgpr_budget=None):
    """调用 inspect_so.py --json，返回解析后的 dict。"""
    if not INSPECT_SO.is_file():
        raise RuntimeError("找不到 %s" % INSPECT_SO)
    cmd = [sys.executable, "-I", str(INSPECT_SO), str(so_path), "--json"]
    if threads is not None:
        cmd += ["--threads", str(threads)]
    if lds is not None:
        cmd += ["--lds", str(lds)]
    if vgpr_budget is not None:
        cmd += ["--vgpr-budget", str(vgpr_budget)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except OSError as e:
        raise RuntimeError("无法运行 inspect_so.py：%s" % e)
    if proc.returncode != 0:
        raise RuntimeError("inspect_so.py 退出 %d：%s"
                           % (proc.returncode, (proc.stderr or proc.stdout).strip()))
    try:
        return json.loads(proc.stdout)
    except ValueError as e:
        raise RuntimeError("inspect_so.py 输出不是 JSON：%s" % e)


def pick_kernel(kernels, name=None):
    """从 inspect_so 的 kernels 里选一个；多个且未指定就报错，不猜。"""
    names = [k.get("name", "?") for k in kernels]
    if name:
        exact = [k for k in kernels if k.get("name") == name]
        if len(exact) == 1:
            return exact[0]
        sub = [k for k in kernels if name in (k.get("name") or "")]
        if len(sub) == 1:
            return sub[0]
        if sub:
            raise ValueError("--kernel %r 有歧义，匹配 %d 个：%s"
                             % (name, len(sub), ", ".join(n for n in names if name in n)))
        raise ValueError("--kernel %r 不在 .so 里；有：%s" % (name, ", ".join(names)))
    if len(kernels) == 1:
        return kernels[0]
    raise ValueError(".so 里有 %d 个 kernel，必须用 --kernel 指定：%s"
                     % (len(kernels), ", ".join(names)))


def _set_unknown(plc, unk, field, reason, force=False, only_if_absent=True):
    """写 null + 原因；已有非 null 值且没 --force 时不动它。

    only_if_absent=True 时，「已有非 null 值就不动」是第二道锁——
    用于 grid：它无法从 .so 推出，是纯人填量，--force 也不许把它抹成 null。
    """
    if plc.get(field) is not None:
        if only_if_absent or not force:
            return False
    plc[field] = None
    unk[field] = reason
    return True


def fill_record(rec, row, threads=None, wave_size=None, num_cu=CU_COUNT,
                force=False):
    """把 inspect_so 的一行 kernel 信息填进一条 record。

    返回 (problems, changes)。只填能从 .so 真实得到的量：
      vgpr / lds_per_cta 总能填；ctas_per_cu 需要实际 launch 线程数；
      waves_per_cu 还需要 wavefront size（gfx9 = 64，但必须显式给）；
      grid 无法从 .so 得到 → null + 原因，留给人填。
    """
    probs = []
    changes = []
    if not isinstance(rec, dict):
        return [_err("record", "不是 JSON 对象")], changes
    if "placement" not in rec or rec["placement"] is None:
        rec["placement"] = {}
        changes.append(("placement", None, {}))
    plc = rec["placement"]
    if not isinstance(plc, dict):
        return [_err("placement", "不是对象，拒绝覆盖")], changes
    unk = rec.get("placement_unknown")
    if not isinstance(unk, dict):
        unk = {}
        rec["placement_unknown"] = unk

    def put(field, value):
        old = plc.get(field, _MISSING)
        if old is not _MISSING and old is not None and not force:
            if old != value:
                probs.append(_warn(
                    "placement.%s" % field,
                    "已有值 %r 与 .so 推出的 %r 不同；保留原值（--force 覆盖）"
                    % (old, value)))
            return
        plc[field] = value
        unk.pop(field, None)
        changes.append((field, None if old is _MISSING else old, value))

    vgpr = row.get("vgpr")
    if not _is_int(vgpr) or vgpr < 1:
        probs.append(_err(
            "placement.vgpr",
            "inspect_so 给出的 VGPR=%r（≤0）；无法用它算占用率，"
            "拒绝填 0（0 会静默变成无限 CTA）" % (vgpr,)))
        return probs, changes
    put("vgpr", vgpr)

    lds_used = row.get("lds_used")
    if _is_int(lds_used) and lds_used >= 0:
        put("lds_per_cta", lds_used)
    else:
        probs.append(_err("placement.lds_per_cta",
                          "inspect_so 给出的 lds_used=%r 非法" % (lds_used,)))

    # grid 永远拿不到；且它是纯人填量——即使 --force 也不许把已填的值抹成 null
    if plc.get("grid") is None or "grid" not in plc:
        _set_unknown(plc, unk, "grid",
                     ".so 里不含 launch grid；必须人填（整数或各维 CTA 数列表）",
                     only_if_absent=True)

    if threads is None:
        _set_unknown(
            plc, unk, "ctas_per_cu",
            "缺少实际 launch 线程数（--threads）；inspect_so 的 "
            "max_flat_workgroup_size=%r 不等于 launch 线程数，用它算占用率会错"
            % (row.get("max_wg"),), force=force)
        _set_unknown(
            plc, unk, "waves_per_cu",
            "缺少实际 launch 线程数与 wavefront size（--threads/--wave-size）",
            force=force)
        return probs, changes

    ctas = row.get("cta_per_cu")
    if not _is_int(ctas) or ctas < 1:
        probs.append(_err(
            "placement.ctas_per_cu",
            "inspect_so 算出的 CTA/CU=%r（≤0）：VGPR*threads 已超过寄存器预算，"
            "kernel 无法以该形状 launch" % (ctas,)))
        return probs, changes

    if ctas > num_cu:
        _set_unknown(
            plc, unk, "ctas_per_cu",
            "inspect_so 的资源上限 CTA/CU=%d 超过 CU 数 %d，不可能是实际驻留；"
            "该公式缺 max-waves/CU 与每 CTA wave 数，需人填实测值"
            % (ctas, num_cu), force=force)
        _set_unknown(
            plc, unk, "waves_per_cu",
            "ctas_per_cu 未知（inspect_so 上限 %d 不可信），waves 无法推" % ctas,
            force=force)
        return probs, changes

    put("ctas_per_cu", ctas)

    if wave_size is None:
        _set_unknown(
            plc, unk, "waves_per_cu",
            "缺 wavefront size（--wave-size；gfx9 为 64）。waves_per_cu = "
            "ctas_per_cu × ceil(threads/wave_size)，必须显式给出",
            force=force)
        return probs, changes
    if wave_size < 1:
        probs.append(_err("--wave-size", "必须 ≥1"))
        return probs, changes

    threads_used = row.get("threads")
    if not _is_int(threads_used) or threads_used < 1:
        probs.append(_err("placement.waves_per_cu",
                          "inspect_so 给出的 threads=%r 非法" % (threads_used,)))
        return probs, changes
    waves_per_cta = int(math.ceil(float(threads_used) / wave_size))
    put("waves_per_cu", ctas * waves_per_cta)
    return probs, changes


def cmd_fill(results_path, so_path, kernel=None, threads=None, lds=None,
             wave_size=None, num_cu=CU_COUNT, force=False, dry_run=False,
             as_json=False):
    doc = load_document(results_path)
    try:
        data = read_so(so_path, threads=threads, lds=lds)
    except RuntimeError as e:
        _die("%s" % e)
    kernels = data.get("kernels") or []
    if not kernels:
        _die("%s 里没解析到 kernel" % so_path)
    try:
        row = pick_kernel(kernels, kernel)
    except ValueError as e:
        _die("%s" % e)

    if not isinstance(doc, dict) or not isinstance(doc.get("records"), list):
        _die("%s 缺少 records 数组" % results_path)

    probs = []
    all_changes = []
    for i, rec in enumerate(doc["records"]):
        rp, ch = fill_record(rec, row, threads=threads, wave_size=wave_size,
                             num_cu=num_cu, force=force)
        for p in rp:
            p.where = "records[%d](%s).%s" % (
                i, rec.get("variant") if isinstance(rec, dict) else "?", p.where)
        probs.extend(rp)
        all_changes.append((i,
                            rec.get("variant") if isinstance(rec, dict) else None,
                            ch))

    vprobs, summary = validate_document(
        doc, results_path, num_cu=num_cu)
    probs.extend(vprobs)

    errs = [p for p in probs if p.severity == "error"]
    wrote = False
    if not dry_run:
        if errs:
            # 有 error 就不落盘：半填入的文件会被下游当成「已经填过」。
            print("未写入 %s：补全后仍有 %d 个 error，先修复再重跑"
                  "（--dry-run 可只看会改成什么）" % (results_path, len(errs)))
        else:
            tmp = Path(str(results_path) + ".tmp")
            tmp.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n",
                           encoding="utf-8")
            os.replace(str(tmp), str(results_path))
            wrote = True

    if as_json:
        print(json.dumps({
            "ok": not errs,
            "written": wrote,
            "dry_run": dry_run,
            "so": str(so_path),
            "kernel": row,
            "changes": [{"record": i, "variant": v,
                         "changed": [{"field": f, "old": o, "new": n}
                                     for f, o, n in ch]}
                        for i, v, ch in all_changes],
            "problems": [p.as_dict() for p in probs],
        }, indent=2, ensure_ascii=False))
    else:
        if dry_run:
            print("（--dry-run：不写文件）")
        elif wrote:
            print("已写入 %s" % results_path)
        print(".so kernel=%s VGPR=%s LDS=%s threads=%r CTA/CU=%s"
              % (row.get("name"), row.get("vgpr"), row.get("lds_used"),
                 row.get("threads"), row.get("cta_per_cu")))
        for i, v, ch in all_changes:
            print("records[%d](%s):" % (i, v))
            if not ch:
                print("  （无可填字段）")
            for f, o, n in ch:
                print("  %-13s %r -> %r" % (f, o, n))
        for p in probs:
            print(p)
        print("结果：%s（%d error）" % ("不通过" if errs else "通过", len(errs)))

    return 1 if errs else 0


# ---------------------------------------------------------------- selftest

def selftest():
    """不需要 GPU、不需要文件的自检。"""
    good = {
        "variant": "demo",
        "placement": {"grid": [128, 1], "ctas_per_cu": 4, "vgpr": 96,
                      "lds_per_cta": 16384, "waves_per_cu": 16},
        "placement_unknown": {},
    }
    p, info = validate_record(good, 0)
    assert not [x for x in p if x.severity == "error"], p
    assert not p, p
    assert info["grid_ctas"] == 128
    print("[ok] 合法完整 record 通过（grid=[128,1] → 128 CTA，4 CTA/CU × 4 wave = 16）")

    import copy

    def only_errors(rec):
        return [x for x in validate_record(rec, 0)[0] if x.severity == "error"]

    def only_warns(rec):
        return [x for x in validate_record(rec, 0)[0] if x.severity == "warning"]

    bad = copy.deepcopy(good)
    del bad["placement"]["vgpr"]
    e = only_errors(bad)
    assert e and any("vgpr" in x.where for x in e), e
    print("[ok] 缺 vgpr 键 → 报错（与 null 不同：省略字段无法带原因）")

    bad = copy.deepcopy(good)
    bad["placement"]["vgpr"] = 0
    e = only_errors(bad)
    assert e and any("vgpr=0" in x.message for x in e), e
    print("[ok] vgpr=0 → 报错（0 不是『没填』，会让占用率变无限 CTA）")

    bad = copy.deepcopy(good)
    bad["placement"] = None
    assert only_errors(bad)
    bad = copy.deepcopy(good)
    del bad["placement"]
    assert only_errors(bad)
    print("[ok] placement 为 null / 缺失 → 报错")

    bad = copy.deepcopy(good)
    bad["placement"]["vgprs"] = 96
    e = only_errors(bad)
    assert any("vgprs" in x.where for x in e), e
    print("[ok] placement 里未知字段（vgprs）→ 报错，拼写错误不静默通过")

    bad = copy.deepcopy(good)
    bad["placement"]["vgpr"] = None
    assert only_errors(bad)
    bad["placement_unknown"] = {"vgpr": "编译器版本未知，.so 已被覆盖"}
    e, info = validate_record(bad, 0)
    assert not [x for x in e if x.severity == "error"], e
    assert info["unknown"] == ["vgpr"], info
    print("[ok] null 无原因 → 报错；null 带原因 → 记为 unknown（不冒充 0）")

    bad = copy.deepcopy(good)
    bad["placement"]["lds_per_cta"] = 0
    assert not validate_record(bad, 0)[0]
    print("[ok] lds_per_cta=0 合法（真的不用 LDS），与 vgpr=0 区别对待")

    bad = copy.deepcopy(good)
    bad["placement"]["ctas_per_cu"] = 128
    w = only_warns(bad)
    assert any("num_cu" in x.message for x in w), w
    print("[ok] ctas_per_cu=128 > 64 CU → 警告")

    bad = copy.deepcopy(good)
    bad["placement"]["waves_per_cu"] = 15
    w = only_warns(bad)
    assert any("整数倍" in x.message for x in w), w
    print("[ok] waves_per_cu=15 与 ctas_per_cu=4 不自洽 → 警告")

    bad = copy.deepcopy(good)
    bad["placement"]["grid"] = 136
    w = only_warns(bad)
    assert any("整数倍" in x.message for x in w), w
    bad = copy.deepcopy(good)
    bad["placement"]["grid"] = 128
    assert not [x for x in only_warns(bad) if "整数倍" in x.message]
    print("[ok] grid=136（非 64 倍数）→ 不均衡警告；128 → 不警告（hbird 反例）")

    bad = copy.deepcopy(good)
    bad["placement"]["grid"] = [0]
    assert only_errors(bad)
    bad["placement"]["grid"] = 8192.0
    assert only_errors(bad)
    print("[ok] grid 元素 ≤0 / 浮点数 → 报错（必须是整数或整数列表）")

    # fill_record 纯逻辑（不调 .so）
    row = {"name": "k", "vgpr": 96, "lds_used": 8192, "max_wg": 256,
           "threads": 256, "cta_per_cu": 4}
    rec = {"variant": "demo", "placement_unknown": {}}
    rp, ch = fill_record(rec, row, threads=256, wave_size=64, num_cu=64)
    assert not [x for x in rp if x.severity == "error"], rp
    assert rec["placement"]["vgpr"] == 96
    assert rec["placement"]["ctas_per_cu"] == 4
    assert rec["placement"]["waves_per_cu"] == 16       # 4 CTA × 4 wave
    assert rec["placement"]["grid"] is None
    assert "grid" in rec["placement_unknown"]
    assert not [x for x in validate_record(rec, 0)[0] if x.severity == "error"]
    print("[ok] fill_record：VGPR/LDS/CTA/CU/waves 自动填；grid 记 null + 原因")

    rec = {"variant": "demo", "placement_unknown": {}}
    fill_record(rec, dict(row, cta_per_cu=384), threads=256, wave_size=64)
    assert rec["placement"]["ctas_per_cu"] is None
    assert "ctas_per_cu" in rec["placement_unknown"]
    print("[ok] fill_record：CTA/CU=384 > 64 → 判为不可信，写 null + 原因")

    rec = {"variant": "demo", "placement_unknown": {}}
    rp, _ = fill_record(rec, row, threads=None, wave_size=None)
    assert rec["placement"]["vgpr"] == 96
    assert rec["placement"]["ctas_per_cu"] is None
    assert "threads" in rec["placement_unknown"]["ctas_per_cu"]
    print("[ok] fill_record：不给 --threads 时不拿 max_flat_workgroup_size 硬算占用率")

    print("\nselftest PASS")
    return 0


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="results json 的 placement 字段校验器（只校验，不解释性能）")
    # 文件用位置参数，`--check` 才是开关：这样 `--check --strict a.json`
    # 与 `--check a.json --strict` 都成立（nargs='+' 会吞掉紧邻的选项名）。
    ap.add_argument("--check", action="store_true",
                    help="校验一批 results json（文件跟在后面）；缺 placement 则非 0 退出")
    ap.add_argument("files", nargs="*", metavar="FILE", help="--check 的输入文件")
    ap.add_argument("--fill-from-so", metavar="RESULTS_JSON",
                    help="用 inspect_so.py 的元数据补全 placement（会写文件，除非 --dry-run）")
    ap.add_argument("--so", metavar="PATH.so", help="--fill-from-so 的编译产物")
    ap.add_argument("--kernel", help=".so 里有多个 kernel 时必须指定")
    ap.add_argument("--threads", type=int, help="实际 launch 的每 CTA 线程数")
    ap.add_argument("--lds", type=int, help="每 CTA 动态 LDS 字节数（extern __shared__）")
    ap.add_argument("--wave-size", type=int, help="wavefront size（gfx9 为 64）")
    ap.add_argument("--num-cu", type=int, default=CU_COUNT,
                    help="CU 数（默认 %d，本机 gfx938）" % CU_COUNT)
    ap.add_argument("--force", action="store_true",
                    help="--fill-from-so 覆盖已有非 null 值（grid 除外，它是人填量）")
    ap.add_argument("--dry-run", action="store_true", help="只打印，不写文件")
    ap.add_argument("--strict", action="store_true", help="警告也导致非 0 退出")
    ap.add_argument("--require-known", action="store_true",
                    help="null（unknown）也算失败，要求实测值")
    ap.add_argument("--json", action="store_true", help="机器可读输出")
    ap.add_argument("--selftest", action="store_true", help="无需 GPU/文件的自检")
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest()
    if a.fill_from_so:
        if not a.so:
            ap.error("--fill-from-so 需要 --so")
        if a.files:
            ap.error("不要同时给位置文件；--fill-from-so 只接收 --fill-from-so 指定的那个文件")
        return cmd_fill(a.fill_from_so, a.so, kernel=a.kernel, threads=a.threads,
                        lds=a.lds, wave_size=a.wave_size, num_cu=a.num_cu,
                        force=a.force, dry_run=a.dry_run, as_json=a.json)
    if a.check:
        if not a.files:
            ap.error("--check 需要至少一个文件")
        return cmd_check(a.files, num_cu=a.num_cu, strict=a.strict,
                         require_known=a.require_known, as_json=a.json)
    if a.files:
        ap.error("给了文件但没给 --check（不知道要对它们做什么）")
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
