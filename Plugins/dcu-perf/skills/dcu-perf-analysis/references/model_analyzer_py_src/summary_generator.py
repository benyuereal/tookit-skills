

import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter


def _get_layer_multiplier(module_name: str, num_dense_layers: int, num_moe_layers: int,
                          csv_layer_count: int = 0) -> int:
    """根据 CSV layer_count 列或 module 名称返回该模块的层数乘数。
    优先使用 CSV 中的 layer_count（PP 模式下为 stage 内实际层数），
    否则按模块名推断（兼容旧 CSV 格式）。
    """
    if csv_layer_count > 0:
        return csv_layer_count
    name = module_name.lower()
    if "dense_layers" in name:
        return num_dense_layers
    if "moe_layers" in name:
        return num_moe_layers
    if "nextn_moe" in name:
        return 1
    if "prefill_chunk" in name or "prefill_remainder" in name:
        return 1
    return 1


def _safe_float(value, default=0.0):
    """安全地将值转换为float，处理None、空字符串或无效值"""
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def _compute_summary(rows, num_dense_layers, num_moe_layers):
    """从 raw rows 计算 summary 统计"""
    stats = defaultdict(lambda: {
        "total_latency_ns": 0.0,
        "total_flops": 0.0,
        "call_count": 0,
        "latencies": [],
        "bandwidths": [],
        "tflops_list": [],
        "total_bytes": 0.0,
        "perf_sources": set(),
        "bound_types": set(),
        "arithmetic_intensities": [],
    })
    stage_total_latency = defaultdict(float)

    for row in rows:
        stage = row["stage"]
        module = row["module"]
        op_type = row["op_type"]
        latency_us = _safe_float(row["latency_us"])
        latency_ns = latency_us * 1000.0  # convert to ns for internal calculation
        tflops = _safe_float(row["tflops"])
        bandwidth_gbs = _safe_float(row["bandwidth(GB/s)"])
        total_flops = _safe_float(row["total_flops"])
        total_bytes = _safe_float(row["total_bytes"])

        perf_source = row.get("perf_source", "unknown")
        bound_type = row.get("bound_type", "unknown")
        try:
            arith_intensity = float(row.get("arithmetic_intensity", 0))
        except (ValueError, TypeError):
            arith_intensity = 0.0

        try:
            csv_layer_count = int(row.get("layer_count", 0))
        except (ValueError, TypeError):
            csv_layer_count = 0
        multiplier = _get_layer_multiplier(module, num_dense_layers, num_moe_layers,
                                           csv_layer_count)
        key = (stage, op_type)
        s = stats[key]
        s["perf_sources"].add(perf_source)
        s["bound_types"].add(bound_type)
        s["total_latency_ns"] += latency_ns * multiplier
        s["total_flops"] += total_flops * multiplier
        s["call_count"] += multiplier
        s["latencies"].append(latency_ns)
        s["bandwidths"].append(bandwidth_gbs)
        s["tflops_list"].append(tflops)
        s["total_bytes"] += total_bytes * multiplier
        s["arithmetic_intensities"].append(arith_intensity)
        stage_total_latency[stage] += latency_ns * multiplier

    sorted_keys = sorted(stats.keys(), key=lambda k: (k[0], -stats[k]["total_latency_ns"]))
    return stats, stage_total_latency, sorted_keys


def generate_excel(
    raw_csv_path: str,
    excel_path: str,
    num_dense_layers: int = 3,
    num_moe_layers: int = 58,
) -> None:
    """
    从 raw CSV 生成 Excel 文件，包含 raw_data 和 summary 两个 sheet

    Args:
        raw_csv_path: 输入的 performance_results.csv 路径
        excel_path: 输出的 .xlsx 路径
        num_dense_layers: Dense 层数量
        num_moe_layers: MoE 层数量
    """
    rows = []
    with open(raw_csv_path, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    if not rows:
        print("Warning: Raw CSV is empty, no Excel generated.")
        return

    # perf_source 列默认隐藏(发布版本不展示); 设环境变量
    # MODEL_ANALYZER_SHOW_PERF_SOURCE=1 (或 true/yes/on) 才显示
    show_perf_source = os.environ.get(
        "MODEL_ANALYZER_SHOW_PERF_SOURCE", ""
    ).strip().lower() in ("1", "true", "yes", "on")

    stats, stage_total_latency, sorted_keys = _compute_summary(rows, num_dense_layers, num_moe_layers)

    header_font = Font(bold=True, size=11)
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    header_font_white = Font(bold=True, size=11, color="FFFFFF")
    thin_border = Border(
        left=Side(style="thin"), right=Side(style="thin"),
        top=Side(style="thin"), bottom=Side(style="thin"),
    )

    wb = Workbook()

    ws_summary = wb.active
    ws_summary.title = "summary"

    summary_header = [
        "stage", "op_type", "perf_source", "bound_type", "call_count",
        "total_latency_ns", "total_latency_ms", "pct_of_stage",
        "avg_latency_ns", "avg_bandwidth_gbs", "avg_tflops",
        "total_flops", "total_bytes", "avg_arithmetic_intensity",
    ]
    if not show_perf_source:
        summary_header.remove("perf_source")

    for col_idx, col_name in enumerate(summary_header, 1):
        cell = ws_summary.cell(row=1, column=col_idx, value=col_name)
        cell.font = header_font_white
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")
        cell.border = thin_border

    for row_idx, key in enumerate(sorted_keys, 2):
        stage, op_type = key
        s = stats[key]
        total_lat = s["total_latency_ns"]
        stage_lat = stage_total_latency[stage]
        pct = (total_lat / stage_lat * 100) if stage_lat > 0 else 0
        avg_latency = sum(s["latencies"]) / len(s["latencies"]) if s["latencies"] else 0
        avg_bw = sum(s["bandwidths"]) / len(s["bandwidths"]) if s["bandwidths"] else 0
        avg_tflops = sum(s["tflops_list"]) / len(s["tflops_list"]) if s["tflops_list"] else 0

        perf_source_str = ",".join(sorted(s["perf_sources"]))
        bound_type_str = ",".join(sorted(s["bound_types"]))
        avg_arith_intensity = (sum(s["arithmetic_intensities"]) / len(s["arithmetic_intensities"])
                               if s["arithmetic_intensities"] else 0)
        row_data = [
            stage, op_type, perf_source_str, bound_type_str, s["call_count"],
            round(total_lat, 2), round(total_lat / 1e6, 4), round(pct, 2),
            round(avg_latency, 2), round(avg_bw, 2), round(avg_tflops, 2),
            round(s["total_flops"]), round(s["total_bytes"]),
            round(avg_arith_intensity, 4),
        ]
        if not show_perf_source:
            del row_data[2]  # 与 summary_header 去掉 perf_source 对齐
        for col_idx, val in enumerate(row_data, 1):
            cell = ws_summary.cell(row=row_idx, column=col_idx, value=val)
            cell.border = thin_border

    for col_idx in range(1, len(summary_header) + 1):
        col_letter = get_column_letter(col_idx)
        max_len = len(summary_header[col_idx - 1])
        for row in ws_summary.iter_rows(min_row=2, max_row=ws_summary.max_row, min_col=col_idx, max_col=col_idx):
            for cell in row:
                if cell.value is not None:
                    max_len = max(max_len, len(str(cell.value)))
        ws_summary.column_dimensions[col_letter].width = min(max_len + 3, 30)

    ws_raw = wb.create_sheet(title="raw_data")

    raw_fieldnames = list(fieldnames)
    has_layer_count = "layer_count" in fieldnames
    if not has_layer_count:
        op_type_idx = fieldnames.index("op_type") if "op_type" in fieldnames else 3
        raw_fieldnames.insert(op_type_idx + 1, "count")
    # perf_source 列默认隐藏(同 summary sheet, 由环境变量控制)
    if not show_perf_source and "perf_source" in raw_fieldnames:
        raw_fieldnames.remove("perf_source")

    for col_idx, col_name in enumerate(raw_fieldnames, 1):
        cell = ws_raw.cell(row=1, column=col_idx, value=col_name)
        cell.font = header_font_white
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")
        cell.border = thin_border

    numeric_cols = {
        "latency_us", "eval_time_us",
        "total_flops", "tflops",
        "tensor_core_flops", "tensor_core_tflops",
        "vector_flops", "vector_tflops",
        "sfu_flops", "sfu_tflops",
        "bandwidth(GB/s)",
        "bytes_read", "bytes_written", "total_bytes",
        "arithmetic_intensity",
    }

    for row_idx, row in enumerate(rows, 2):
        module_name = row.get("module", "")
        try:
            csv_layer_count = int(row.get("layer_count", 0))
        except (ValueError, TypeError):
            csv_layer_count = 0
        count = _get_layer_multiplier(module_name, num_dense_layers, num_moe_layers,
                                      csv_layer_count)
        for col_idx, col_name in enumerate(raw_fieldnames, 1):
            if col_name == "count" and not has_layer_count:
                val = count
            elif col_name == "layer_count":
                val = count  # 使用计算后的 count 覆盖原始值
            else:
                val = row[col_name]
                if col_name in numeric_cols:
                    try:
                        val = float(val)
                    except (ValueError, TypeError):
                        pass
            cell = ws_raw.cell(row=row_idx, column=col_idx, value=val)
            cell.border = thin_border

    for col_idx in range(1, len(raw_fieldnames) + 1):
        col_letter = get_column_letter(col_idx)
        max_len = len(raw_fieldnames[col_idx - 1])
        for row in ws_raw.iter_rows(min_row=2, max_row=min(ws_raw.max_row, 10), min_col=col_idx, max_col=col_idx):
            for cell in row:
                if cell.value is not None:
                    max_len = max(max_len, len(str(cell.value)[:30]))
        ws_raw.column_dimensions[col_letter].width = min(max_len + 3, 25)

    wb.save(excel_path)
    print(f"Excel 已保存: {excel_path}")
    print(f"  - sheet 'summary': {len(sorted_keys)} 条汇总记录")
    print(f"  - sheet 'raw_data': {len(rows)} 条算子详细记录")

    _print_summary_table(sorted_keys, stats, stage_total_latency, show_perf_source)


def generate_summary(
    raw_csv_path: str,
    summary_csv_path: str,
    num_dense_layers: int = 3,
    num_moe_layers: int = 58,
) -> None:
    """兼容旧接口，现在内部调用 generate_excel"""
    excel_path = summary_csv_path.replace(".csv", ".xlsx")
    if not excel_path.endswith(".xlsx"):
        excel_path = str(Path(summary_csv_path).parent / "results.xlsx")
    generate_excel(raw_csv_path, excel_path, num_dense_layers, num_moe_layers)


def _print_summary_table(sorted_keys, stats, stage_total_latency, show_perf_source=False):
    """控制台打印 summary 表格 (source 列默认隐藏, 与 Excel 同受环境变量控制)"""
    current_stage = None
    for key in sorted_keys:
        stage, op_type = key
        s = stats[key]

        if stage != current_stage:
            current_stage = stage
            stage_lat = stage_total_latency[stage]
            print(f"\n  [{stage}] 总延迟: {stage_lat/1e6:.4f} ms")
            source_hdr = f"{'source':<18} " if show_perf_source else ""
            print(f"  {'op_type':<25} {source_hdr}{'calls':>6} {'total_lat(ms)':>14} {'pct':>8} "
                  f"{'avg_lat(ns)':>14} {'avg_bw(GB/s)':>14} {'avg_tflops':>12}")
            print(f"  {'-'*(113 if show_perf_source else 95)}")

        total_lat = s["total_latency_ns"]
        stage_lat = stage_total_latency[stage]
        pct = (total_lat / stage_lat * 100) if stage_lat > 0 else 0
        avg_latency = sum(s["latencies"]) / len(s["latencies"]) if s["latencies"] else 0
        avg_bw = sum(s["bandwidths"]) / len(s["bandwidths"]) if s["bandwidths"] else 0
        avg_tflops = sum(s["tflops_list"]) / len(s["tflops_list"]) if s["tflops_list"] else 0

        source_col = f"{','.join(sorted(s['perf_sources'])):<18} " if show_perf_source else ""
        print(f"  {op_type:<25} {source_col}{s['call_count']:>6} {total_lat/1e6:>14.4f} {pct:>7.2f}% "
              f"{avg_latency:>14.2f} {avg_bw:>14.2f} {avg_tflops:>12.2f}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        raw_path = sys.argv[1]
        excel_path = sys.argv[2] if len(sys.argv) >= 3 else "results.xlsx"
        generate_excel(raw_path, excel_path)
    else:
        print("Usage: python summary_generator.py <raw_csv_path> [output.xlsx]")
