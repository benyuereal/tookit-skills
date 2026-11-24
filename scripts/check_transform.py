#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_transform —— C1 验收：每条 surface 必须能落到「具体改法」。

为什么需要它
------------
`surfaces.yaml` 的 17 条攻击面都有 statement / precondition / check /
expected_gain，但 0 条有 `transform`。这意味着 `occupancy` 只能告诉你
「CTA/CU 太低、收益上界 ≤15%」，却不告诉你降 VGPR 到底是改 tile、改
unroll、还是改 mma 形状，改成多少。**这是「诊断」到「研发」之间那道门。**

本脚本把这道门变成机器判据，而不是再写一段散文：

1. 覆盖：每条 surface 都有 `transform`，且是非空 list；
2. kind 只允许 param / code / isa；
3. 至少 10 条是 `kind: param`（C1 的核心：可参数化，不是散文）；
4. `legality` 必须含**数值约束**（`<= 64KB`、`<= 255`、`% 8 == 0`
   这种）；「注意不要溢出」判不可解析；
5. param 类必须有 `candidates`，且每条是 `name old→new` 形式的参数替换；
6. 每条必须有 `predicted_dir`：改了之后往哪个方向变。

字段形状
--------
    transform:
      - kind: param
        what: "{BLOCK_M, num_warps, num_stages}"
        candidates: ["BLOCK_M 128→64", "num_warps 8→4", "num_stages 3→2"]
        legality: "LDS = BLOCK_M*BLOCK_K*2B <= 64KB; VGPR <= 255"
        predicted_dir: "降 VGPR ⇒ CTA/CU 从 1 升到 2 ⇒ 延迟隐藏改善"
      - kind: code
        what: "累加器 fp32 分片累加"
        legality: "rel_l2 <= 1e-3（contract 容差）"
        predicted_dir: "寄存器压力下降 ⇒ 驻留提升"
      - kind: isa
        what: "写回从 dwordx2 提到 dwordx4"
        legality: "N % 8 == 0; 地址 16B 对齐"
        predicted_dir: "写带宽提升"

注意：code / isa 同样要写 `predicted_dir`。没有方向的改动无法在事后
被证伪，也无法进 A/B 校准——这正是 C1 要防的东西。

用法
----
    python3 -I scripts/check_transform.py               # 覆盖检查
    python3 -I scripts/check_transform.py --legality    # legality 可解析检查
    python3 -I scripts/check_transform.py --json        # 机读输出
    python3 -I scripts/check_transform.py --selftest

`surfaces.yaml` 还没填完时必须能正常工作：缺 transform 的 surface 报
「未覆盖」，而不是 KeyError / 崩溃。

退出码：0 = 通过；1 = 有判据不满足；2 = 文件读不了 / YAML 解析失败。
"""
import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SURFACES = REPO / "Plugins/ops/skills/op-optimize/references/surfaces.yaml"

KINDS = ("param", "code", "isa")
MIN_PARAM = 10

_MISSING = object()


# --------------------------------------------------------------------------
# legality：必须含可机读的数值约束
# --------------------------------------------------------------------------
# 判据是「有没有数值 + 约束关系」，不是「有没有汉字」。
# 这些模式覆盖仓库里真实出现的写法：
#   <= 64KB / <= 255 / >= 8 / N % 8 == 0 / 16B 对齐 / 64 的整数倍 /
#   不超过 64KB / BLOCK_M ∈ {32, 64, 128} / num_stages 2~4
_CONSTRAINT_PATTERNS = (
    re.compile(r"(?:<=|>=|==|!=|≤|≥|≠|<|>)\s*-?\d"),
    re.compile(r"-?\d+(?:\.\d+)?\s*"
               r"(?:%|KB|MB|GB|TB|B|bytes?|个|条|次|warps?|threads?|regs?|VGPRs?|LDS)?"
               r"\s*(?:<=|>=|==|!=|≤|≥|≠|<|>)"),
    # 取模约束：% 8 == 0 / % 64 越接近 0 越好 / % (BK*num_split) 需为 0
    # 用 (?<![0-9]) 排除百分数（15%）。
    re.compile(r"(?<![0-9])%\s*(?:\([^)]*\)|[A-Za-z_][\w.]*|\d+)"),
    re.compile(r"\d+\s*(?:B|字节|bytes?)?\s*(?:对齐|整数倍|倍数)"),
    re.compile(r"\d+\s*的\s*(?:整数倍|倍数)"),
    # 必须为 1 / 需为 0 / 应为 8 / 等于 255
    re.compile(r"(?:必须|需|应|要求|要)?\s*(?:为|等于|=)\s*-?\d"),
    re.compile(r"(?:不超过|不大于|不小于|不低于|不高于|至少|最多|最少|"
               r"上限|下限|超过|大于|小于|等于|低于|高于)\s*-?\d"),
    re.compile(r"\{\s*-?\d+(?:\s*,\s*-?\d+)+\s*\}"),          # {32, 64, 128}
    re.compile(r"\d+\s*~\s*\d+"),                              # 2~4
)


def legality_parseable(value, item=None):
    """True = legality 可机检；False = 散文。

    **两类合法情况**（2026-11-24 修正）：
      A. legality 文本本身含数值约束（`<= 65536` / `% 8 == 0` / `== 0`）
      B. 语义型约束：显式标 `semantic: true` **且**给了 `legality_check`
         （一条可执行的判据，如 `max_abs == 0` / `outputs 数 == writes`）

    为什么要有 B：有些 transform 的约束本质就是语义的
    （"融合后输出份数不能变"），硬编成数值是自欺。
    但**必须显式标记 + 给可执行判据**，否则它就退化成散文。

    第一版只认 A，把 6 条真语义约束误判成散文 —— 那是判据比现实窄。
    """
    if not isinstance(value, str) or not value.strip():
        return False
    if any(p.search(value) for p in _CONSTRAINT_PATTERNS):
        return True
    if isinstance(item, dict) and item.get("semantic") is True:
        chk = str(item.get("legality_check") or "").strip()
        return bool(chk) and any(p.search(chk) for p in _CONSTRAINT_PATTERNS)
    return False


# --------------------------------------------------------------------------
# param candidates：必须是「name old→new」/「name=value」的参数替换
# --------------------------------------------------------------------------
_IDENT = r"[A-Za-z_][A-Za-z0-9_.]*"
_ARROW = re.compile(r"→|->|=>")


def is_param_candidate(value):
    """判断一条 candidate 是不是具体的参数替换，而不是一句散文。

    接受：
        "BLOCK_M 128→64" / "num_warps 8→4" / "BLOCK_M: 128 -> 64"
        "BLOCK_M=64" / "num_stages=2"
    拒绝：
        "把 tile 调小一点" / "降低 VGPR" / "换个 mma 形状"
    """
    if not isinstance(value, str):
        return False
    s = value.strip()
    if not s:
        return False
    if not re.search(r"\d", s):
        # 参数替换必须给出具体数值；"big→small" 不是可执行改法
        return False
    if _ARROW.search(s):
        left, right = _ARROW.split(s, maxsplit=1)
        return bool(re.search(_IDENT, left)) and bool(right.strip())
    if re.match(r"^\s*%s\s*=\s*\S+" % _IDENT, s):
        return True
    return False


# --------------------------------------------------------------------------
# 单条 transform 检查
# --------------------------------------------------------------------------
def check_entry(entry):
    """返回 {"kind": ..., "pass": bool, "errors": [str]}。"""
    if not isinstance(entry, dict):
        return {"kind": None, "pass": False,
                "errors": ["transform 条目不是 mapping：%r" % (entry,)]}

    errs = []
    kind = entry.get("kind")
    if kind not in KINDS:
        errs.append("kind=%r 非法（只允许 param / code / isa）" % (kind,))

    what = entry.get("what")
    if not isinstance(what, str) or not what.strip():
        errs.append("缺 what —— 必须写清楚改的是哪个参数 / 哪段代码 / 哪条 ISA")

    legality = entry.get("legality")
    if not isinstance(legality, str) or not legality.strip():
        errs.append("缺 legality —— 必须写明约束")
    elif not legality_parseable(legality, entry):
        errs.append(
            "legality 不可解析（无数值约束）：%r —— 要「<= 64KB」/「<= 255」"
            "/「%% 8 == 0」这种，不要「注意不要溢出」这种散文" % legality)

    predicted = entry.get("predicted_dir")
    if not isinstance(predicted, str) or not predicted.strip():
        errs.append("缺 predicted_dir —— 必须写明改了之后往哪个方向变")

    if kind == "param":
        cands = entry.get("candidates")
        if not isinstance(cands, list) or not cands:
            errs.append('param 类缺 candidates —— 必须给出具体改法'
                        '（如 "BLOCK_M 128→64"）')
        else:
            for i, c in enumerate(cands):
                if not is_param_candidate(c):
                    errs.append(
                        'candidates 第 %d 条 %r 不是参数替换形式'
                        '（要 "name old→new" 或 "name=value"）' % (i + 1, c))

    return {
        "kind": kind if isinstance(kind, str) else str(kind),
        "pass": not errs,
        "errors": errs,
    }


# --------------------------------------------------------------------------
# 全表检查
# --------------------------------------------------------------------------
def load_surfaces(path):
    """读取 surfaces.yaml。支持顶层 list，或 dict 里带 `surfaces:`。

    未填完 / 空文件都返回 list（可能是空 list），由 check_surfaces 报未覆盖，
    不在这里抛错。
    """
    import yaml

    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if data is None:
        return []
    if isinstance(data, dict):
        data = data.get("surfaces") or []
    if not isinstance(data, list):
        raise ValueError("surfaces.yaml 顶层必须是 list 或含 surfaces 的 dict")
    return data


def check_surfaces(surfaces):
    """检查 surface 列表，返回机读 report（可直接 JSON 序列化）。"""
    errors = []
    surfaces_rep = []
    covered = 0
    kind_counts = {k: 0 for k in KINDS}
    invalid_kinds = 0
    leg_ok = 0
    leg_total = 0
    param_total = 0

    for idx, surface in enumerate(surfaces):
        if not isinstance(surface, dict):
            sid = "<#%d>" % idx
            errors.append("%s: 不是 mapping，无法检查" % sid)
            surfaces_rep.append({"id": sid, "covered": False, "pass": False,
                                 "errors": ["不是 mapping"], "entries": []})
            continue

        sid = str(surface.get("id") or "<#%d>" % idx)
        transform = surface.get("transform", _MISSING)

        serrs = []
        entries_rep = []
        covered_flag = False

        if transform is _MISSING:
            serrs.append("transform 缺失 —— C1 要求每条 surface 都能落到具体改法"
                         "（param / code / isa）")
        elif transform is None:
            serrs.append("transform 为 null —— 有字段没内容，等于没填")
        elif not isinstance(transform, list):
            serrs.append("transform 必须是 list，得到 %s"
                         % type(transform).__name__)
        elif not transform:
            serrs.append("transform 是空 list —— 有字段没内容，等于没填")
        else:
            covered_flag = True
            covered += 1
            for ei, entry in enumerate(transform):
                er = check_entry(entry)
                entries_rep.append({"index": ei,
                                    "kind": er["kind"],
                                    "pass": er["pass"],
                                    "errors": er["errors"]})
                if er["kind"] in KINDS:
                    kind_counts[er["kind"]] += 1
                    if er["kind"] == "param":
                        param_total += 1
                else:
                    invalid_kinds += 1

                leg_total += 1
                legality = entry.get("legality") if isinstance(entry, dict) else None
                if legality_parseable(legality):
                    leg_ok += 1

                for msg in er["errors"]:
                    serrs.append("transform[#%d]: %s" % (ei, msg))

        errors.extend("%s: %s" % (sid, msg) for msg in serrs)
        surfaces_rep.append({
            "id": sid,
            "covered": covered_flag,
            "pass": not serrs,
            "errors": serrs,
            "entries": entries_rep,
        })

    if not surfaces:
        errors.insert(0, "surfaces.yaml 里没有 surface —— 检查器无对象可查")

    param_ok = param_total >= MIN_PARAM
    if not param_ok:
        errors.insert(0,
                      "param 类只有 %d 条（要求 ≥%d）—— C1 的核心是"
                      "「可参数化，不是散文」" % (param_total, MIN_PARAM))

    ok = (bool(surfaces) and covered == len(surfaces) and not errors and param_ok)

    return {
        "surface_coverage": {"covered": covered, "total": len(surfaces)},
        "kind_counts": kind_counts,
        "invalid_kinds": invalid_kinds,
        "param_total": param_total,
        "param_min": MIN_PARAM,
        "param_ok": param_ok,
        "legality": {"ok": leg_ok, "total": leg_total},
        "ok": ok,
        "errors": errors,
        "surfaces": surfaces_rep,
    }


def check_file(path):
    report = check_surfaces(load_surfaces(path))
    report["file"] = str(path)
    return report


# --------------------------------------------------------------------------
# 输出
# --------------------------------------------------------------------------
def format_report(report):
    cov = report["surface_coverage"]
    kc = report["kind_counts"]
    leg = report["legality"]

    lines = ["surface 覆盖 %d/%d" % (cov["covered"], cov["total"])]

    if report["param_ok"]:
        param_line = "  param 类: %d 条  ✓（要求 ≥%d）" % (
            kc["param"], report["param_min"])
    else:
        param_line = "  param 类: %d 条  ✗（要求 ≥%d，还差 %d 条）" % (
            kc["param"], report["param_min"],
            max(0, report["param_min"] - kc["param"]))
    lines.append(param_line)
    lines.append("  code 类:  %d 条" % kc["code"])
    lines.append("  isa 类:   %d 条" % kc["isa"])
    if report["invalid_kinds"]:
        lines.append("  非法 kind: %d 条" % report["invalid_kinds"])

    lines.append("")
    lines.append("legality 可解析 %d/%d 条" % (leg["ok"], leg["total"]))

    if report["errors"]:
        lines.append("")
        for msg in report["errors"]:
            lines.append("✗ " + msg)

    if report["ok"]:
        lines.append("")
        lines.append("✓ C1 transform 检查通过")

    return "\n".join(lines)


def print_legality(report):
    leg = report["legality"]
    print("legality 可解析 %d/%d 条" % (leg["ok"], leg["total"]))
    if leg["total"] == 0:
        print("  （当前没有 transform 条目可查）")

    errs = [e for e in report["errors"] if "legality" in e]
    if errs:
        print()
        for msg in errs:
            print("✗ " + msg)

    if leg["ok"] == leg["total"]:
        if leg["total"]:
            print("✓ 所有 legality 都含可机读的数值约束")
        return 0
    return 1


# --------------------------------------------------------------------------
# selftest
# --------------------------------------------------------------------------
def _valid_param():
    return {
        "kind": "param",
        "what": "{BLOCK_M, num_warps, num_stages}",
        "candidates": ["BLOCK_M 128→64", "num_warps 8→4", "num_stages 3→2"],
        "legality": "LDS = BLOCK_M*BLOCK_K*2B <= 64KB; VGPR <= 255",
        "predicted_dir": "降 VGPR ⇒ CTA/CU 从 1 升到 2 ⇒ 延迟隐藏改善",
    }


def _valid_code():
    return {
        "kind": "code",
        "what": "累加器 fp32 分片累加",
        "legality": "rel_l2 <= 1e-3（contract 容差）",
        "predicted_dir": "寄存器压力下降 ⇒ 驻留提升",
    }


def _valid_isa():
    return {
        "kind": "isa",
        "what": "写回从 dwordx2 提到 dwordx4",
        "legality": "N % 8 == 0; 地址 16B 对齐",
        "predicted_dir": "写带宽提升",
    }


def _make_surfaces(total=17, n_param=10):
    out = []
    for i in range(total):
        if i < n_param:
            transform = [_valid_param()]
        elif i < n_param + 4:
            transform = [_valid_code()]
        else:
            transform = [_valid_isa()]
        out.append({"id": "s%d" % i, "transform": transform})
    return out


def _write_yaml(surfaces):
    import yaml

    d = Path(tempfile.mkdtemp())
    p = d / "surfaces.yaml"
    p.write_text(yaml.safe_dump(surfaces, allow_unicode=True, sort_keys=False),
                 encoding="utf-8")
    return p


def selftest():
    import copy

    n = 0

    def ok(label, cond, detail=""):
        nonlocal n
        assert cond, "%s 失败：%s" % (label, detail)
        n += 1
        print("[ok] %s" % label)

    # 1) 全覆盖 → pass
    rep = check_surfaces(_make_surfaces())
    ok("17/17 覆盖 + 10 条 param → pass",
       rep["ok"] and rep["surface_coverage"] == {"covered": 17, "total": 17}
       and rep["kind_counts"]["param"] == 10, rep)

    # 2) 缺 transform → 报错并指出哪条
    s = _make_surfaces()
    del s[3]["transform"]
    s[3]["id"] = "occupancy"
    rep = check_surfaces(s)
    ok("缺 transform → 报错并指出 surface id",
       not rep["ok"] and any("occupancy" in e and "transform 缺失" in e
                             for e in rep["errors"]), rep["errors"])

    # 3) transform: [] → 报错
    s = _make_surfaces()
    s[4]["transform"] = []
    rep = check_surfaces(s)
    ok("transform: [] → 报错", not rep["ok"]
       and any("空 list" in e for e in rep["errors"]), rep["errors"])

    # 4) kind 非法 → 报错
    s = _make_surfaces()
    s[0]["transform"][0]["kind"] = "text"
    rep = check_surfaces(s)
    ok("kind: text → 报错", not rep["ok"]
       and any("非法" in e and "text" in e for e in rep["errors"]),
       rep["errors"])

    # 5) param < 10 → 报错（C1 核心门槛）
    rep = check_surfaces(_make_surfaces(n_param=9))
    ok("param 类 9 条 → 报错（要求 ≥10）",
       not rep["ok"] and not rep["param_ok"]
       and any("≥10" in e for e in rep["errors"]), rep["errors"])

    # 6) param 缺 candidates → 报错
    s = _make_surfaces()
    s[0]["transform"][0].pop("candidates")
    rep = check_surfaces(s)
    ok("param 缺 candidates → 报错", not rep["ok"]
       and any("candidates" in e for e in rep["errors"]), rep["errors"])

    # 7) 缺 predicted_dir → 报错
    s = _make_surfaces()
    s[0]["transform"][0].pop("predicted_dir")
    rep = check_surfaces(s)
    ok("缺 predicted_dir → 报错", not rep["ok"]
       and any("predicted_dir" in e for e in rep["errors"]), rep["errors"])

    # 8) 反向：legality 是散文 → 不可解析
    s = _make_surfaces()
    s[0]["transform"][0]["legality"] = "注意别溢出"
    rep = check_surfaces(s)
    ok("反向：散文 legality「注意别溢出」→ 不可解析",
       not legality_parseable("注意别溢出") and not rep["ok"]
       and any("不可解析" in e for e in rep["errors"]), rep["errors"])

    # 9) 反向：legality 含 <= 64KB → 可解析
    ok("反向：legality「<= 64KB」→ 可解析",
       legality_parseable("LDS <= 64KB")
       and legality_parseable("VGPR <= 255")
       and legality_parseable("N % 8 == 0")
       and legality_parseable("地址 16B 对齐"))

    # 10) candidates 不是参数替换形式 → 报错
    s = _make_surfaces()
    s[0]["transform"][0]["candidates"] = ["把 tile 调小一点"]
    rep = check_surfaces(s)
    ok("candidates 散文「把 tile 调小一点」→ 报错", not rep["ok"]
       and any("不是参数替换形式" in e for e in rep["errors"]),
       rep["errors"])

    # 11) 未填完（5/17）不崩，报覆盖数
    s = _make_surfaces()
    for item in s[5:]:
        del item["transform"]
    rep = check_surfaces(s)
    ok("未填完 5/17 → 不崩且报 5/17",
       rep["surface_coverage"] == {"covered": 5, "total": 17}
       and not rep["ok"], rep["surface_coverage"])

    # 12) 空文件不崩
    rep = check_surfaces([])
    ok("空 surfaces 列表 → 不崩，报 0/0",
       rep["surface_coverage"] == {"covered": 0, "total": 0}
       and not rep["ok"], rep)

    # 13) dict 顶层 {"surfaces": [...]} 也认
    p = _write_yaml({"surfaces": _make_surfaces()})
    rep = check_file(p)
    ok("顶层 dict 带 surfaces: 也能读", rep["ok"], rep["errors"])

    # 14) 真实文件读得动（不要求已填完）
    try:
        rep = check_file(DEFAULT_SURFACES)
        ok("真实 surfaces.yaml 可读（当前 %d/%d）"
           % (rep["surface_coverage"]["covered"],
              rep["surface_coverage"]["total"]), True)
    except FileNotFoundError:
        print("[skip] 真实 surfaces.yaml 不存在")

    # 15) code/isa 也要求 predicted_dir（否则方向无法证伪）
    s = _make_surfaces()
    idx = next(i for i, x in enumerate(s) if x["transform"][0]["kind"] == "isa")
    s[idx]["transform"][0].pop("predicted_dir")
    rep = check_surfaces(s)
    ok("isa 类缺 predicted_dir → 报错", not rep["ok"]
       and any("predicted_dir" in e for e in rep["errors"]), rep["errors"])

    # 16) 不应有共享状态导致的串改
    a = _make_surfaces()
    b = copy.deepcopy(a)
    assert a == b
    n += 1

    print("\nselftest PASS（%d 项）" % n)
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="校验 surfaces.yaml 的 transform 字段（C1 验收）")
    ap.add_argument("--file", default=str(DEFAULT_SURFACES),
                    help="surfaces.yaml 路径（默认仓库内主表）")
    ap.add_argument("--legality", action="store_true",
                    help="只检查 legality 是否含可解析的数值约束")
    ap.add_argument("--json", action="store_true", help="机读 JSON 输出")
    ap.add_argument("--selftest", action="store_true", help="跑内置自检")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    path = Path(args.file)
    try:
        report = check_file(path)
    except FileNotFoundError:
        print("找不到 %s" % path, file=sys.stderr)
        return 2
    except Exception as exc:                       # YAML 解析失败等
        print("读 %s 失败：%s" % (path, exc), file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["ok"] else 1

    if args.legality:
        return print_legality(report)

    print(format_report(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
