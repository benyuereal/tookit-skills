#!/usr/bin/env python3
import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract rocBLAS GEMM M/N/K from log and export CSV."
    )
    p.add_argument("-i", "--input", default="rocblas.txt", help="Input log file")
    p.add_argument("--detail-out", default="rocblas_detail.csv", help="Detail CSV output")
    p.add_argument("--summary-out", default="rocblas_summary.csv", help="Summary CSV output")
    p.add_argument("--tp", default="", help="Optional tp value, e.g. 1")
    return p.parse_args()


def extract_from_bench_line(line: str):
    ta = re.search(r"--transposeA\s+([TN])\b", line, re.I)
    tb = re.search(r"--transposeB\s+([TN])\b", line, re.I)
    m = re.search(r"(?:^|\s)-m\s+(\d+)\b", line)
    n = re.search(r"(?:^|\s)-n\s+(\d+)\b", line)
    k = re.search(r"(?:^|\s)-k\s+(\d+)\b", line)
    if not (ta and tb and m and n and k):
        return None
    return {
        "计算类型": f"{ta.group(1).upper()}{tb.group(1).upper()}",
        "M": int(m.group(1)),
        "N": int(n.group(1)),
        "K": int(k.group(1)),
    }


def extract_from_api_line(line: str):
    m = re.search(
        r"rocblas_gemm(?:_strided_batched)?_ex,([TN]),([TN]),(\d+),(\d+),(\d+),",
        line,
        re.I,
    )
    if not m:
        return None
    return {
        "计算类型": f"{m.group(1).upper()}{m.group(2).upper()}",
        "M": int(m.group(3)),
        "N": int(m.group(4)),
        "K": int(m.group(5)),
    }


def extract_records(text: str):
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        rec = None
        if "rocblas-bench" in line:
            rec = extract_from_bench_line(line)
        elif "rocblas_gemm_ex," in line or "rocblas_gemm_strided_batched_ex," in line:
            rec = extract_from_api_line(line)

        if rec is not None:
            records.append(rec)
    return records


def write_detail_csv(records, output_path: Path, tp_value: str):
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["序号", "计算类型", "M", "N", "K", "tp"])
        for i, r in enumerate(records, start=1):
            w.writerow([i, r["计算类型"], r["M"], r["N"], r["K"], tp_value])


def write_summary_csv(records, output_path: Path, tp_value: str):
    # 保留全部 M/K 组合，只把 N 做范围聚合
    groups = defaultdict(list)
    for r in records:
        key = (r["计算类型"], r["M"], r["K"], tp_value)
        groups[key].append(r["N"])

    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        # 标题行
        w.writerow(["rocmblas gemm size", "", "", "", ""])
        # 表头
        w.writerow(["计算类型", "M", "N_range", "K", "tp"])

        # 数据：tp 只在第一行显示，后续留空（视觉上接近合并）
        first = True
        for (calc_type, m, k, tp), n_list in sorted(
            groups.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3])
        ):
            n_min = min(n_list)
            n_max = max(n_list)
            n_range = f"{n_min}-{n_max}" if n_min != n_max else str(n_min)

            tp_cell = tp if first else ""
            w.writerow([calc_type, m, n_range, k, tp_cell])
            first = False


def main():
    args = parse_args()
    in_path = Path(args.input)
    if not in_path.exists():
        print(f"[ERROR] Input file not found: {in_path}", file=sys.stderr)
        return 1

    text = in_path.read_text(encoding="utf-8", errors="ignore")
    records = extract_records(text)
    if not records:
        print("[ERROR] No valid rocBLAS GEMM records found.", file=sys.stderr)
        return 2

    detail_out = Path(args.detail_out)
    summary_out = Path(args.summary_out)
    write_detail_csv(records, detail_out, args.tp)
    write_summary_csv(records, summary_out, args.tp)

    print(f"[OK] Parsed records: {len(records)}")
    print(f"[OK] Detail CSV: {detail_out}")
    print(f"[OK] Summary CSV: {summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
