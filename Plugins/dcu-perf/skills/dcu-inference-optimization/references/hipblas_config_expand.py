#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""hipblas_config_expand.py —— hipBLASLt 调优日志 → 按 n 全覆盖的 (m,k,n)→solution_index 映射表

做三件事（与 hipblas-cofig-expanded-28col.txt 的生成规则一致）：

  1. 删列：默认删除 8 个布局列
       lda, stride_a, ldb, stride_b, ldc, stride_c, ldd, stride_d
     这些列在原文件里 66/66 行都满足 lda=ldb=k、ldc=ldd=m、stride_a=m*k、
     stride_b=n*k、stride_c=stride_d=n*m，是 m,n,k 的机械推导量，不是独立参数。

  2. 补 n：在每个 (m,k) 分组内，把 n 在 [--min-n, --max-n]（默认 256..2048）补齐。
     原有实测 n 的行原样保留（只删列），新行按"实测锚点"推导 solution_index。

  3. 推导规则（--rule，默认 nearest）：
       nearest : 取 n 最近的实测锚点；并列时按 --tie（默认 lower=取较小 n）
       floor   : 取 ≤ n 的最大锚点（无则用最小锚点）
       ceil    : 取 ≥ n 的最小锚点（无则用最大锚点）
     n < 最小锚点 → 用最小锚点的 solution_index；n > 最大锚点 → 用最大锚点的。
     （nearest 规则下这两条是自然结果；floor/ceil 也遵循同一"贴边"约定。）

  4. 新行只改 n 与 solution_index：其余列逐字节沿用"来源行"（提供该 solution_index
     的那个实测 n 的行）。因此新行的 hipblaslt-Gflops / us 仍是来源 n 的实测值。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter, OrderedDict

PROG = "hipblas_config_expand.py"
VERSION = "1.1"

# 默认删除的列（布局推导列）
DEFAULT_DROP = ["lda", "stride_a", "ldb", "stride_b", "ldc", "stride_c", "ldd", "stride_d"]
# 允许在同一个 (m,k) 分组内变化的"测量列"（新行沿用来源行的值，不参与一致性校验）
DEFAULT_ALLOW_VARYING = ["hipblaslt-Gflops", "us"]


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
class Row:
    """一行数据。raw 保留原始行（缩进在内），fields 为去掉首尾空白后按 ',' 切分的字段。"""

    __slots__ = ("raw", "indent", "fields")

    def __init__(self, raw: str, indent: str, fields: list[str]):
        self.raw = raw
        self.indent = indent
        self.fields = fields


def eprint(*args) -> None:
    print(*args, file=sys.stderr)


def detect_eol(text: str) -> str:
    m = re.search(r"\r\n|\n|\r", text)
    return m.group(0) if m else "\n"


def read_text(path: str, encoding: str) -> tuple[str, bool]:
    with open(path, "rb") as fh:
        raw = fh.read()
    has_bom = raw.startswith(b"\xef\xbb\xbf")
    enc = encoding
    if has_bom and enc.lower().replace("-", "").replace("_", "") in ("utf8", "utf8sig"):
        enc = "utf-8-sig"
    return raw.decode(enc), has_bom


def is_data_line(stripped: str, ncol: int) -> bool:
    if not stripped or "," not in stripped:
        return False
    f = stripped.split(",")
    return len(f) == ncol and f[0] in ("T", "N")


def parse_log(text: str):
    """返回 (header: list[str], preamble: list[str], rows: list[Row])"""
    lines = re.split(r"\r\n|\n|\r", text)
    while lines and lines[-1] == "":
        lines.pop()

    hidx = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("transA,"):
            hidx = i
            break
    if hidx is None:  # 兜底：以第一个数据行的同字段数上一行作为表头
        for i, ln in enumerate(lines):
            f = ln.strip().split(",")
            if len(f) >= 5 and f[0] in ("T", "N"):
                for j in range(i - 1, -1, -1):
                    s = lines[j].strip()
                    if s and len(s.split(",")) == len(f):
                        hidx = j
                        break
                break
    if hidx is None:
        raise ValueError("找不到表头行（应以 'transA,' 开头）与数据行，文件格式不符合 hipBLASLt 日志。")

    header = [c.strip() for c in lines[hidx].strip().split(",")]
    ncol = len(header)
    preamble = lines[:hidx]
    rows: list[Row] = []
    for ln in lines[hidx + 1:]:
        s = ln.strip()
        if not is_data_line(s, ncol):
            continue  # 重复的表头行 / 空行 / 其他行一律忽略，输出时重新生成
        rows.append(Row(raw=ln, indent=ln[: len(ln) - len(ln.lstrip())], fields=s.split(",")))
    if not rows:
        raise ValueError("没有解析到任何数据行。")
    return header, preamble, rows


def resolve_columns(tokens: list[str], header: list[str], what: str,
                    missing_out: list[str] | None = None) -> list[int]:
    """把列名或 0 基下标解析成下标列表。

    missing_out 非 None 时，找不到的列名只记录、不抛错（供默认删列清单降级为告警）。
    """
    idx: list[int] = []
    for tok in tokens:
        tok = tok.strip()
        if tok == "":
            continue
        if re.fullmatch(r"\d+", tok):
            i = int(tok)
            if i >= len(header):
                raise ValueError(f"{what}: 列下标 {i} 超出范围（共 {len(header)} 列）。")
            idx.append(i)
        else:
            if tok not in header:
                if missing_out is not None:
                    missing_out.append(tok)
                    continue
                near = [h for h in header if tok.lower() in h.lower()]
                hint = f"（相近列名: {', '.join(near)}）" if near else ""
                raise ValueError(f"{what}: 找不到列名 '{tok}'{hint}")
            idx.append(header.index(tok))
    return idx


def pick_source(n: int, anchors: list[int], rule: str, tie: str) -> int:
    if rule == "floor":
        le = [a for a in anchors if a <= n]
        return max(le) if le else anchors[0]
    if rule == "ceil":
        ge = [a for a in anchors if a >= n]
        return min(ge) if ge else anchors[-1]
    if tie == "upper":
        return min(anchors, key=lambda a: (abs(n - a), -a))
    return min(anchors, key=lambda a: (abs(n - a), a))


def segments(pairs: list[tuple[int, str]]) -> list[str]:
    """把 [(n, value)] 压成 ['[a-b] value', ...]（连续同值合并）。"""
    out: list[str] = []
    start = prev = None
    cur = None
    for n, v in pairs:
        if cur is None:
            cur, start = v, n
        elif v != cur:
            out.append(f"[{start}-{prev}] {cur}")
            cur, start = v, n
        prev = n
    if cur is not None:
        out.append(f"[{start}-{prev}] {cur}")
    return out


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=PROG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "把 hipBLASLt 调优日志扩写成按 n 全覆盖的 (m,k,n) → solution_index 映射表。\n"
            "原实测 n 的行原样保留（仅删列）；在 [--min-n, --max-n] 内补齐的新行，\n"
            "只改 n 与 solution_index，其余列沿用提供该 solution_index 的来源行。"
        ),
        epilog="""\
典型用法
  # 1) 默认：删 8 个布局列 + 补 n=256..2048（每个整数）→ 28 列映射表
  python hipblas_config_expand.py hipblas-cofig.txt -o hipblas-cofig-expanded-28col.txt

  # 2) 先看不写：打印统计与分段表
  python hipblas_config_expand.py hipblas-cofig.txt --dry-run --summary

  # 3) 换推导规则：区间向下取整（分桶单调）
  python hipblas_config_expand.py hipblas-cofig.txt --rule floor -o out-floor.txt

  # 4) 只删列、不补 n；或保留全部列
  python hipblas_config_expand.py in.txt --no-expand -o out.txt
  python hipblas_config_expand.py in.txt --drop none -o out.txt

  # 5) 看这个文件有哪些列（列名 / 0 基下标都可用于 --drop）
  python hipblas_config_expand.py hipblas-cofig.txt --list-columns

输出格式
  与输入同构：preamble（如 'Git Version: ...'）+ 每个数据行前一行表头 + 数据行；
  行尾符、BOM 与输入保持一致。

退出码
  0 成功 | 1 运行/校验失败（信息在 stderr） | 2 参数用法错误
""",
    )

    p.add_argument("input", help="输入文件：hipBLASLt 调优日志（表头行以 transA, 开头）")
    p.add_argument("-o", "--output", default=None,
                   help="输出文件；默认在输入名后加 '-processed'（如 hipblas-cofig-processed.txt）")
    p.add_argument("--force", action="store_true",
                   help="允许输出路径与输入相同（原地覆盖）")

    g = p.add_argument_group("删列")
    g.add_argument("--drop", default=",".join(DEFAULT_DROP), metavar="COL[,COL...]",
                   help="要删除的列，列名或 0 基下标，逗号分隔；'none' 表示不删。默认: "
                        + ",".join(DEFAULT_DROP))
    g.add_argument("--allow-varying", default=",".join(DEFAULT_ALLOW_VARYING), metavar="COL[,COL...]",
                   help="允许在同一分组内变化的测量列（豁免一致性校验）。默认: "
                        + ",".join(DEFAULT_ALLOW_VARYING))
    g.add_argument("--list-columns", action="store_true",
                   help="打印输入文件的列名与下标后退出，不写文件")

    g = p.add_argument_group("补 n")
    g.add_argument("--min-n", type=int, default=256, metavar="N", help="补齐区间下界（含）。默认 256")
    g.add_argument("--max-n", type=int, default=2048, metavar="N", help="补齐区间上界（含）。默认 2048")
    g.add_argument("--step", type=int, default=1, metavar="K", help="补齐步长。默认 1（每个整数）")
    g.add_argument("--no-expand", action="store_true", help="只删列，不补 n")
    g.add_argument("--rule", choices=("nearest", "floor", "ceil"), default="nearest",
                   help="新 n 的 solution_index 推导规则。默认 nearest")
    g.add_argument("--tie", choices=("lower", "upper"), default="lower",
                   help="nearest 且左右等距时的取向。默认 lower（取较小 n）")

    g = p.add_argument_group("分组与列名")
    g.add_argument("--key", default="m,k", metavar="COL[,COL...]",
                   help="分组键（同一 GEMM 的恒定列）。默认 m,k")
    g.add_argument("--n-column", default="n", help="n 所在列名。默认 n")
    g.add_argument("--solution-column", default="solution_index", help="默认 solution_index")

    g = p.add_argument_group("输出与检查")
    g.add_argument("--encoding", default="utf-8", help="输入编码。默认 utf-8（自动识别 BOM）")
    g.add_argument("--lenient", action="store_true",
                   help="分组一致性校验失败时只告警、继续处理（默认报错退出）")
    g.add_argument("--summary", action="store_true", help="打印每个分组的 solution 分段表")
    g.add_argument("--dry-run", action="store_true", help="只统计与校验，不写输出文件")
    g.add_argument("-q", "--quiet", action="store_true", help="不打印统计")
    p.add_argument("--version", action="version", version=f"{PROG} {VERSION}")
    return p


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)

    # ---- 读入与解析 ----
    try:
        text, has_bom = read_text(args.input, args.encoding)
        header, preamble, rows = parse_log(text)
    except (OSError, ValueError, UnicodeDecodeError) as e:
        eprint(f"[error] 读取/解析失败: {e}")
        return 1

    eol = detect_eol(text)
    ncol = len(header)

    if args.list_columns:
        print(f"# {args.input}  {len(rows)} 数据行, {ncol} 列")
        for i, name in enumerate(header):
            print(f"{i:>3}  {name}")
        return 0

    # ---- 定位列 ----
    try:
        n_idx = header.index(args.n_column) if args.n_column in header else None
        if n_idx is None:
            raise ValueError(f"找不到 n 列 '{args.n_column}'；可用列: {', '.join(header)}")
        sol_idx = header.index(args.solution_column) if args.solution_column in header else None
        if sol_idx is None:
            raise ValueError(f"找不到 solution 列 '{args.solution_column}'")
        key_idx = resolve_columns([c for c in args.key.split(",")], header, "--key")
        if not key_idx:
            raise ValueError("--key 不能为空")
        if n_idx in key_idx or sol_idx in key_idx:
            raise ValueError("--key 不能包含 n 列或 solution 列")
        missing_drop: list[str] = []
        drop_idx = set() if args.drop.strip().lower() == "none" else set(
            resolve_columns(args.drop.split(","), header, "--drop",
                            missing_out=missing_drop if args.drop == ",".join(DEFAULT_DROP) else None))
        if missing_drop:
            eprint(f"[warn] --drop 默认清单里这些列在输入中不存在，已跳过: {', '.join(missing_drop)}"
                   "（输入可能已是处理过的文件）")
        for special, nm in ((n_idx, "n"), (sol_idx, "solution")):
            if special in drop_idx:
                raise ValueError(f"不允许删除 {nm} 列（'{header[special]}'）")
        allow_idx = set(resolve_columns(args.allow_varying.split(","), header, "--allow-varying"))
    except ValueError as e:
        eprint(f"[error] {e}")
        return 2

    keep_idx = [i for i in range(ncol) if i not in drop_idx]

    # ---- 解析每行的数值 ----
    parsed: list[tuple[tuple[str, ...], int, Row]] = []
    for r in rows:
        try:
            n = int(r.fields[n_idx])
        except ValueError:
            eprint(f"[error] 数据行的 n 不是整数: {r.raw.strip()[:80]}...")
            return 1
        parsed.append((tuple(r.fields[i] for i in key_idx), n, r))

    groups: "OrderedDict[tuple[str, ...], dict[int, Row]]" = OrderedDict()
    dup = []
    for key, n, row in parsed:
        d = groups.setdefault(key, {})
        if n in d:
            dup.append((key, n))
        else:
            d[n] = row
    if dup:
        for key, n in dup[:5]:
            eprint(f"[error] 分组 {key} 内 n={n} 出现多次，无法确定来源行。")
        eprint("[error] 请把区分这些行的列（如 batch_count）加入 --key。")
        return 1

    # ---- 一致性校验：分组内除 n / solution / 测量列 / 待删列外必须恒定 ----
    # 只校验"保留列"：待删列（如由 n 推导的 stride_*）变化与否不影响结果。
    violations: list[str] = []
    exempt = {n_idx, sol_idx} | allow_idx | drop_idx
    for key, d in groups.items():
        rows_in = list(d.values())
        for i in range(ncol):
            if i in exempt:
                continue
            vals = {r.fields[i] for r in rows_in}
            if len(vals) > 1:
                shown = ", ".join(sorted(vals)[:6])
                violations.append(f"  分组 {key}: 列 '{header[i]}' 有 {len(vals)} 种取值 ({shown})")
    if violations:
        eprint("[warn] 同一分组内存在非 n 的差异列：")
        for v in violations[:20]:
            eprint(v)
        eprint("       → 建议把这些列并入 --key，或加 --allow-varying 豁免。")
        if not args.lenient:
            eprint("[error] 已中止（加 --lenient 可继续）。")
            return 1

    # ---- 每组锚点 ----
    lo, hi = args.min_n, args.max_n
    if args.step < 1:
        eprint("[error] --step 必须 >= 1")
        return 2
    if lo > hi:
        eprint(f"[error] --min-n({lo}) 不能大于 --max-n({hi})")
        return 2
    fill = [] if args.no_expand else list(range(lo, hi + 1, args.step))
    if fill and len(fill) > 200000:
        eprint(f"[error] 待补齐的 n 有 {len(fill)} 个，过大（上限 200000）。请调大 --step 或缩小区间。")
        return 2

    anchors_of: dict[tuple[str, ...], list[int]] = {}
    for key, d in groups.items():
        a = sorted(n for n in d if lo <= n <= hi)
        if not a:
            eprint(f"[error] 分组 {key} 在 [{lo}, {hi}] 内没有任何实测 n，无法推导 solution_index。")
            eprint("        请调整 --min-n/--max-n，或用 --no-expand。")
            return 1
        anchors_of[key] = a

    # ---- 生成输出 ----
    all_n = sorted({n for _, n, _ in parsed} | set(fill))
    out: list[str] = list(preamble)
    header_line = ",".join(header[i] for i in keep_idx)
    stats = Counter()
    usage: Counter[int] = Counter()
    segs: dict[tuple[str, ...], list[tuple[int, str]]] = {k: [] for k in groups}

    for n in all_n:
        for key, d in groups.items():
            if n in d:                     # 实测行：原样保留（仅删列）
                row = d[n]
                src_n = n
                stats["kept"] += 1
            elif fill and lo <= n <= hi:   # 新行：只改 n 与 solution_index
                src_n = pick_source(n, anchors_of[key], args.rule, args.tie)
                tpl = d[src_n]
                fields = list(tpl.fields)
                fields[n_idx] = str(n)
                fields[sol_idx] = tpl.fields[sol_idx]
                row = Row(raw="", indent=tpl.indent, fields=fields)
                stats["new"] += 1
                usage[src_n] += 1
            else:
                continue
            out.append(header_line)
            out.append(row.indent + ",".join(row.fields[i] for i in keep_idx))
            segs[key].append((n, row.fields[sol_idx]))

    payload = eol.join(out) + eol

    # ---- 报告 ----
    out_path = args.output
    if out_path is None and not args.dry_run:
        stem, ext = os.path.splitext(args.input)
        out_path = f"{stem}-processed{ext}"
    if out_path and os.path.abspath(out_path) == os.path.abspath(args.input) and not args.force:
        eprint("[error] 输出路径与输入相同；确要原地覆盖请加 --force。")
        return 2

    if not args.quiet:
        print(f"输入      : {args.input}  ({len(rows)} 数据行, {ncol} 列, {len(groups)} 个分组, EOL={eol!r})")
        print(f"列        : {ncol} → {len(keep_idx)}"
              + (f"   删除 {len(drop_idx)} 列: " + ", ".join(header[i] for i in sorted(drop_idx)) if drop_idx else "   （不删列）"))
        if header[n_idx] != "n" or header[sol_idx] != "solution_index":
            print(f"关键列    : n=#{n_idx}('{header[n_idx]}')  solution=#{sol_idx}('{header[sol_idx]}')")
        if fill:
            out_of_range = sorted({n for _, n, _ in parsed if not (lo <= n <= hi)})
            print(f"n 覆盖    : {lo}..{hi} step {args.step} ({len(fill)} 个值)"
                  f" + 区间外实测 n {out_of_range if out_of_range else '无'}")
            print(f"推导规则  : {args.rule}" + (f" (tie={args.tie})" if args.rule == "nearest" else ""))
        else:
            print("n 覆盖    : --no-expand（只删列）")
        print(f"行数      : {stats['kept']} 保留 + {stats['new']} 新增 = {stats['kept'] + stats['new']}"
              f"   (表头行 {stats['kept'] + stats['new']} 行，总 {len(out)} 行)")
        if usage:
            print("来源 n 分布: " + "  ".join(f"{k}:{v}" for k, v in sorted(usage.items())))

        if args.summary:
            print("\n每个分组的 solution_index 分段（补齐区间内）:")
            for key in groups:
                label = "(" + ", ".join(key) + ")"
                seg_list = segments(segs[key])
                print(f"  {label:<16} 锚点={anchors_of[key]}")
                print(f"    " + "  ".join(seg_list))

    if args.dry_run:
        if not args.quiet:
            print("\n[dry-run] 未写文件。")
        return 0

    try:
        with open(out_path, "wb") as fh:
            if has_bom:
                fh.write(b"\xef\xbb\xbf")
            fh.write(payload.encode(args.encoding))
    except OSError as e:
        eprint(f"[error] 写文件失败: {e}")
        return 1

    if not args.quiet:
        size = os.path.getsize(out_path)
        print(f"输出      : {out_path}  ({size:,} 字节, {size / 1048576:.2f} MB)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
