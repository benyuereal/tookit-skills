#!/usr/bin/env python3
import argparse
import csv
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract hipblaslt-bench GEMM M/N/K from log and export report+tuning CSV."
    )
    p.add_argument("-i", "--input", default="hipblas.txt", help="Input log file")
    p.add_argument("--detail-out", default="hipblaslt_detail.csv", help="Detail CSV output")
    p.add_argument("--report-out", default="hipblaslt_report.csv", help="Report CSV output (N_range)")
    p.add_argument("--tuning-out", default="hipblaslt_tuning.csv", help="Tuning CSV output (keep all N + count)")
    p.add_argument("--tp", default="", help="Optional tp value, e.g. 1")
    return p.parse_args()


def extract_records(text: str):
    cmd_pattern = re.compile(r"hipblaslt-bench\b.*?(?=hipblaslt-bench\b|$)", re.S)
    m_pat = re.compile(r"(?:^|\s)-m\s+(\d+)\b")
    n_pat = re.compile(r"(?:^|\s)-n\s+(\d+)\b")
    k_pat = re.compile(r"(?:^|\s)-k\s+(\d+)\b")
    ta_pat = re.compile(r"--transA\s+([TN])\b", re.I)
    tb_pat = re.compile(r"--transB\s+([TN])\b", re.I)

    records = []
    for block in cmd_pattern.findall(text):
        mm = m_pat.search(block)
        nn = n_pat.search(block)
        kk = k_pat.search(block)
        if not (mm and nn and kk):
            continue

        ta = ta_pat.search(block)
        tb = tb_pat.search(block)
        calc_type = f"{ta.group(1).upper()}{tb.group(1).upper()}" if (ta and tb) else ""

        records.append({
            "计算类型": calc_type,
            "M": int(mm.group(1)),
            "N": int(nn.group(1)),
            "K": int(kk.group(1)),
        })
    return records


def write_detail_csv(records, output_path: Path, tp_value: str):
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["序号", "计算类型", "M", "N", "K", "tp"])
        for i, r in enumerate(records, start=1):
            w.writerow([i, r["计算类型"], r["M"], r["N"], r["K"], tp_value])


def write_report_csv(records, output_path: Path, tp_value: str):
    # 报告版：N_range，不要次数
    groups = defaultdict(list)
    for r in records:
        key = (r["计算类型"], r["M"], r["K"], tp_value)
        groups[key].append(r["N"])

    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["hipblaslt gemm size", "", "", "", ""])
        w.writerow(["计算类型", "M", "N_range", "K", "tp"])

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


def write_tuning_csv(records, output_path: Path, tp_value: str):
    # 调优版：保留所有具体N，并统计次数
    counter = Counter((r["计算类型"], r["M"], r["N"], r["K"], tp_value) for r in records)

    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["hipblaslt tuning sizes", "", "", "", "", ""])
        w.writerow(["计算类型", "M", "N", "K", "tp", "次数"])

        for (calc_type, m, n, k, tp), cnt in sorted(
            counter.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3], x[0][4])
        ):
            w.writerow([calc_type, m, n, k, tp, cnt])


def main():
    args = parse_args()
    in_path = Path(args.input)
    if not in_path.exists():
        print(f"[ERROR] Input file not found: {in_path}", file=sys.stderr)
        return 1

    text = in_path.read_text(encoding="utf-8", errors="ignore")
    records = extract_records(text)
    if not records:
        print("[ERROR] No valid hipblaslt-bench GEMM records found.", file=sys.stderr)
        return 2

    detail_out = Path(args.detail_out)
    report_out = Path(args.report_out)
    tuning_out = Path(args.tuning_out)

    write_detail_csv(records, detail_out, args.tp)
    write_report_csv(records, report_out, args.tp)
    write_tuning_csv(records, tuning_out, args.tp)

    print(f"[OK] Parsed records: {len(records)}")
    print(f"[OK] Detail CSV:  {detail_out}")
    print(f"[OK] Report CSV:  {report_out}")
    print(f"[OK] Tuning CSV:  {tuning_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
