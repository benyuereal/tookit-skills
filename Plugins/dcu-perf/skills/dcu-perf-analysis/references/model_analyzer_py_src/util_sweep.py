

import argparse
import csv
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

try:
    from .model_analyzer import (model_analyzer, build_cli_parser,
                                 _MODEL_ANALYZER_CLI_TYPES)
    from .summary_generator import _compute_summary
except ImportError:  # 直接脚本运行
    from model_analyzer import (model_analyzer, build_cli_parser,
                                _MODEL_ANALYZER_CLI_TYPES)
    from summary_generator import _compute_summary


_PER_SCENARIO_COLS = [
    "latency_us", "eval_time_us", "tflops",
    "tensor_core_tflops", "vector_tflops", "sfu_tflops",
    "bandwidth(GB/s)", "perf_source", "bound_type",
]

_NUMERIC_COLS = {
    "latency_us", "eval_time_us", "total_flops", "tflops",
    "tensor_core_flops", "tensor_core_tflops",
    "vector_flops", "vector_tflops", "sfu_flops", "sfu_tflops",
    "bandwidth(GB/s)", "bytes_read", "bytes_written", "total_bytes",
    "arithmetic_intensity",
}

_MODEL_METRICS = [
    ("prefill_latency_ms",  "prefill_ms",     ".4f"),
    ("ttft_ms",             "ttft_ms",        ".4f"),
    ("decode_latency_ms",   "decode_step_ms", ".4f"),
    ("total_time_ms",       "total_ms",       ".2f"),
    ("prefill_tps",         "prefill_tps",    ".2f"),
    ("decode_tps_per_step", "decode_tps",     ".2f"),
    ("overall_tps",         "overall_tps",    ".2f"),
    ("sys_throughput_tps",  "sys_tps",        ".2f"),
    ("bottleneck",          "bottleneck",     "s"),
]


@contextmanager
def _suppress_stdout():
    """fd 级 + sys.stdout 级压制 (覆盖 C++ printf 和 Python print)。"""
    sys.stdout.flush()
    old_py = sys.stdout
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd = os.dup(1)
    try:
        os.dup2(devnull_fd, 1)
        sys.stdout = open(os.devnull, "w")
        yield
    finally:
        sys.stdout.flush()
        sys.stdout.close()
        sys.stdout = old_py
        os.dup2(saved_fd, 1)
        os.close(saved_fd)
        os.close(devnull_fd)


def _fmt_util(v) -> str:
    """利用率值格式化: <0 (未设置) → 'def', 否则去尾零。"""
    if v is None or v < 0:
        return "def"
    return f"{v:g}"


def _scenario_label(sc: dict, idx: int) -> str:
    """场景标签: 优先用显式 label, 否则命名 tc{tc}_vec{vec}_sfu{sfu}_bw{bw}。

    四项利用率全进标签, 否则 tc/bw 相同而 vec/sfu 不同的场景会撞名 ->
    raw_data/summary 的 per-场景列互相覆盖, 分不清是哪组利用率。
    """
    if sc.get("label"):
        return str(sc["label"])
    return (f"tc{_fmt_util(sc.get('tensor_core', -1.0))}"
            f"_vec{_fmt_util(sc.get('vector', -1.0))}"
            f"_sfu{_fmt_util(sc.get('sfu', -1.0))}"
            f"_bw{_fmt_util(sc.get('bandwidth', -1.0))}")


def model_analyzer_util_sweep(
    utilization_scenarios: List[dict],
    output_xlsx: Optional[str] = None,
    quiet_scenarios: bool = True,
    **model_analyzer_kwargs,
) -> dict:
    """
    扫描多个利用率场景, 合并输出单表 + 模型级对比。

    Args:
        utilization_scenarios: 场景列表, 每个场景是 dict, 可含键:
            tensor_core / vector / sfu / bandwidth  (省略 → 回退 device JSON)
            label  (可选, 输出列名; 省略 → 自动 tc{tc}_bw{bw})
            例: [{"tensor_core":0.6,"vector":0.3,"sfu":0.3,"bandwidth":0.7},
                 {"tensor_core":0.7,"vector":0.35,"sfu":0.3,"bandwidth":0.75},
                 {"tensor_core":0.8,"vector":0.4,"sfu":0.3,"bandwidth":0.8}]
        output_xlsx: 合并表输出路径; None → 自动生成到 raw_data/。
        quiet_scenarios: True 时压制每个场景的中间打印, 只保留最终对比输出。
        **model_analyzer_kwargs: 透传给 model_analyzer 的其余参数 (model/gpu/并行配置等)。
            包括 2026-05-22 新增的 `moe_dp_size`, `moe_dense_tp_size`, `moe_a2a_backend`
            (LMSYS 风格部署 + a2a backend 选择, 见集成文档 §4.5).

    Returns:
        dict: {"scenarios": [...每场景 result...], "merged_xlsx": <路径>}
    """
    if not utilization_scenarios:
        raise ValueError("utilization_scenarios 不能为空")

    for k in ("util_tensor_core", "util_vector", "util_sfu", "util_bandwidth",
              "_sweep_label", "generate_csv"):
        model_analyzer_kwargs.pop(k, None)

    scenarios = []
    seen_labels = {}
    print(f"\n{'='*60}")
    print(f"  利用率扫描: 共 {len(utilization_scenarios)} 个场景")
    print(f"{'='*60}")
    for idx, sc in enumerate(utilization_scenarios):
        label = _scenario_label(sc, idx)
        # 防撞名: 标签重复 (含用户显式传重复 label) 时追加 #n, 避免 per-场景列覆盖
        if label in seen_labels:
            seen_labels[label] += 1
            print(f"[util_sweep] 警告: 场景标签 {label!r} 重复, 改名为 "
                  f"{label}#{seen_labels[label]}")
            label = f"{label}#{seen_labels[label]}"
        else:
            seen_labels[label] = 0
        kw = dict(model_analyzer_kwargs)
        kw.update(
            generate_csv=True,
            util_tensor_core=float(sc.get("tensor_core", -1.0)),
            util_vector=float(sc.get("vector", -1.0)),
            util_sfu=float(sc.get("sfu", -1.0)),
            util_bandwidth=float(sc.get("bandwidth", -1.0)),
            _sweep_label=label,
        )
        if quiet_scenarios:
            with _suppress_stdout():
                res = model_analyzer(**kw)
        else:
            res = model_analyzer(**kw)
        scenarios.append({"label": label, "spec": sc, "result": res})

    merged_xlsx = _merge_and_write(scenarios, output_xlsx)

    _print_model_comparison(scenarios)

    for s in scenarios:
        p = s["result"].get("raw_csv_path")
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass

    if merged_xlsx:
        print(f"\n合并表已保存: {merged_xlsx}")
    print(f"{'='*60}\n")

    return {"scenarios": scenarios, "merged_xlsx": merged_xlsx}


def _merge_and_write(scenarios: List[dict], output_xlsx: Optional[str]) -> Optional[str]:
    """读取各场景 raw CSV, 合并成单 xlsx (raw_data / summary / model_comparison)。"""
    valid = [s for s in scenarios
             if s["result"].get("raw_csv_path")
             and os.path.exists(s["result"]["raw_csv_path"])]
    if not valid:
        print("[util_sweep] 无有效 raw CSV (可能所有场景显存不可行), 仅输出模型级对比")
        valid = []

    if output_xlsx is None:
        res0 = scenarios[0]["result"]
        raw0 = res0.get("raw_csv_path")
        out_dir = Path(raw0).parent if raw0 else (Path.cwd() / "raw_data")
        out_dir.mkdir(exist_ok=True)
        base = res0.get("filename_prefix", "model")
        suffix = f"_util{scenarios[0]['label']}"
        if base.endswith(suffix):
            base = base[: -len(suffix)]
        output_xlsx = str(out_dir / f"{base}_utilsweep.xlsx")

    header_font_white = Font(bold=True, size=11, color="FFFFFF")
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    scene_fill = PatternFill(start_color="70AD47", end_color="70AD47", fill_type="solid")
    thin = Border(left=Side(style="thin"), right=Side(style="thin"),
                  top=Side(style="thin"), bottom=Side(style="thin"))

    wb = Workbook()
    ws_cmp = wb.active
    ws_cmp.title = "model_comparison"
    _write_model_comparison_sheet(ws_cmp, scenarios, header_font_white, header_fill, thin)

    if valid:
        _write_raw_sheet(wb, valid, header_font_white, header_fill, scene_fill, thin)
        _write_summary_sheet(wb, valid, header_font_white, header_fill, scene_fill, thin)

    wb.save(output_xlsx)
    return output_xlsx


def _write_raw_sheet(wb, valid, header_font, header_fill, scene_fill, thin):
    """合并 per-op raw_data: 共享列 + 每场景一组时间派生列。按行下标对齐 (算子图与利用率无关)。"""
    per_label_rows = {}
    fieldnames = None
    for s in valid:
        with open(s["result"]["raw_csv_path"], newline="") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
            per_label_rows[s["label"]] = list(reader)

    n = min(len(rows) for rows in per_label_rows.values())
    for s in valid:
        if len(per_label_rows[s["label"]]) != n:
            print(f"[util_sweep] 警告: 场景 {s['label']} 算子行数 "
                  f"{len(per_label_rows[s['label']])} 与基准 {n} 不一致, 按最短对齐")

    per_cols = [c for c in _PER_SCENARIO_COLS if c in fieldnames]
    shared_cols = [c for c in fieldnames if c not in per_cols]
    labels = [s["label"] for s in valid]

    ws = wb.create_sheet(title="raw_data")
    header = list(shared_cols)
    for lab in labels:
        header += [f"{c}@{lab}" for c in per_cols]
    for ci, name in enumerate(header, 1):
        cell = ws.cell(row=1, column=ci, value=name)
        cell.font = header_font
        cell.fill = header_fill if ci <= len(shared_cols) else scene_fill
        cell.alignment = Alignment(horizontal="center")
        cell.border = thin

    base_rows = per_label_rows[labels[0]]
    for i in range(n):
        ci = 1
        for c in shared_cols:
            v = base_rows[i].get(c, "")
            if c in _NUMERIC_COLS:
                try:
                    v = float(v)
                except (ValueError, TypeError):
                    pass
            cell = ws.cell(row=i + 2, column=ci, value=v)
            cell.border = thin
            ci += 1
        for lab in labels:
            row = per_label_rows[lab][i]
            for c in per_cols:
                v = row.get(c, "")
                if c in _NUMERIC_COLS:
                    try:
                        v = float(v)
                    except (ValueError, TypeError):
                        pass
                cell = ws.cell(row=i + 2, column=ci, value=v)
                cell.border = thin
                ci += 1
    _autofit(ws, len(header))


def _write_summary_sheet(wb, valid, header_font, header_fill, scene_fill, thin):
    """合并 summary: 按 (stage, op_type) 聚合, 共享 call_count/flops/bytes + 每场景延迟/算力。"""
    res0 = valid[0]["result"]
    n_dense = int(res0.get("num_dense_layers", 3))
    n_moe = int(res0.get("num_moe_layers", 58))

    per_label_stats = {}
    per_label_stage = {}
    key_order = None
    for s in valid:
        with open(s["result"]["raw_csv_path"], newline="") as f:
            rows = list(csv.DictReader(f))
        stats, stage_total, sorted_keys = _compute_summary(rows, n_dense, n_moe)
        per_label_stats[s["label"]] = stats
        per_label_stage[s["label"]] = stage_total
        if key_order is None:
            key_order = sorted_keys

    labels = [s["label"] for s in valid]
    shared_header = ["stage", "op_type", "call_count", "total_flops", "total_bytes"]
    per_header = ["total_latency_ms", "pct_of_stage", "avg_tflops", "avg_bandwidth_gbs", "perf_source"]

    ws = wb.create_sheet(title="summary")
    header = list(shared_header)
    for lab in labels:
        header += [f"{c}@{lab}" for c in per_header]
    for ci, name in enumerate(header, 1):
        cell = ws.cell(row=1, column=ci, value=name)
        cell.font = header_font
        cell.fill = header_fill if ci <= len(shared_header) else scene_fill
        cell.alignment = Alignment(horizontal="center")
        cell.border = thin

    for ri, key in enumerate(key_order, 2):
        stage, op_type = key
        base_stats = per_label_stats[labels[0]].get(key, {})
        row_vals = [
            stage, op_type,
            base_stats.get("call_count", 0),
            round(base_stats.get("total_flops", 0.0)),
            round(base_stats.get("total_bytes", 0.0)),
        ]
        for lab in labels:
            s = per_label_stats[lab].get(key)
            if s is None:
                row_vals += ["", "", "", "", ""]
                continue
            total_lat = s["total_latency_ns"]
            stage_lat = per_label_stage[lab].get(stage, 0.0)
            pct = (total_lat / stage_lat * 100) if stage_lat > 0 else 0.0
            avg_tflops = (sum(s["tflops_list"]) / len(s["tflops_list"])
                          if s["tflops_list"] else 0.0)
            avg_bw = (sum(s["bandwidths"]) / len(s["bandwidths"])
                      if s["bandwidths"] else 0.0)
            row_vals += [
                round(total_lat / 1e6, 4), round(pct, 2),
                round(avg_tflops, 2), round(avg_bw, 2),
                ",".join(sorted(s["perf_sources"])),
            ]
        for ci, v in enumerate(row_vals, 1):
            cell = ws.cell(row=ri, column=ci, value=v)
            cell.border = thin
    _autofit(ws, len(header))


def _write_model_comparison_sheet(ws, scenarios, header_font, header_fill, thin):
    """模型级对比 sheet: 每个利用率场景一行。"""
    header = ["scenario", "tensor_core", "vector", "sfu", "bandwidth"] + \
             [h for _, h, _ in _MODEL_METRICS]
    for ci, name in enumerate(header, 1):
        cell = ws.cell(row=1, column=ci, value=name)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")
        cell.border = thin

    for ri, s in enumerate(scenarios, 2):
        sc, res = s["spec"], s["result"]
        row = [
            s["label"],
            _fmt_util(sc.get("tensor_core", -1.0)),
            _fmt_util(sc.get("vector", -1.0)),
            _fmt_util(sc.get("sfu", -1.0)),
            _fmt_util(sc.get("bandwidth", -1.0)),
        ]
        for key, _hdr, fmt in _MODEL_METRICS:
            v = res.get(key, "")
            if fmt != "s" and isinstance(v, (int, float)):
                v = round(float(v), 4)
            row.append(v)
        for ci, v in enumerate(row, 1):
            cell = ws.cell(row=ri, column=ci, value=v)
            cell.border = thin
    _autofit(ws, len(header))


def _autofit(ws, ncols):
    for ci in range(1, ncols + 1):
        letter = get_column_letter(ci)
        max_len = 0
        for row in ws.iter_rows(min_row=1, max_row=min(ws.max_row, 30),
                                min_col=ci, max_col=ci):
            for cell in row:
                if cell.value is not None:
                    max_len = max(max_len, len(str(cell.value)[:32]))
        ws.column_dimensions[letter].width = min(max_len + 3, 30)


def _print_model_comparison(scenarios: List[dict]) -> None:
    """控制台打印模型级对比表 (每个利用率场景一行)。"""
    print(f"\n{'='*60}")
    print(f"  模型性能 — 利用率场景对比")
    print(f"{'='*60}")
    hdrs = [h for _, h, _ in _MODEL_METRICS]
    line = f"  {'scenario':<16}" + "".join(f"{h:>15}" for h in hdrs)
    print(line)
    print(f"  {'-' * (16 + 15 * len(hdrs))}")
    for s in scenarios:
        res = s["result"]
        cells = []
        for key, _hdr, fmt in _MODEL_METRICS:
            v = res.get(key)
            if v is None or v == "":
                cells.append(f"{'-':>15}")
            elif fmt == "s":
                cells.append(f"{str(v):>15}")
            else:
                try:
                    cells.append(f"{float(v):>15{fmt}}")
                except (ValueError, TypeError):
                    cells.append(f"{str(v):>15}")
        print(f"  {s['label']:<16}" + "".join(cells))
    print(f"{'='*60}")


# 场景串里 key 的别名 → model_analyzer_util_sweep 认的标准键
_SCENARIO_KEY_ALIAS = {
    "tc": "tensor_core", "tensor_core": "tensor_core",
    "vec": "vector", "vector": "vector", "sfu": "sfu",
    "bw": "bandwidth", "bandwidth": "bandwidth", "label": "label",
}


def _parse_scenarios(spec: str) -> List[dict]:
    """解析 `--scenarios` 字符串 → 利用率场景列表。

    格式: 分号分隔多个场景, 每个场景逗号分隔 `key=val`;
    key 别名: tc/tensor_core, vec/vector, sfu, bw/bandwidth, label。
    省略的 util 项回退 device JSON 默认。
    例: "tc=0.6,vec=0.3,sfu=0.3,bw=0.7; tc=0.8,bw=0.85,label=high_tc"
    """
    scenarios = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        sc = {}
        for kv in chunk.split(","):
            kv = kv.strip()
            if not kv:
                continue
            if "=" not in kv:
                raise argparse.ArgumentTypeError(f"场景项缺少 '=': {kv!r}")
            k, v = kv.split("=", 1)
            std = _SCENARIO_KEY_ALIAS.get(k.strip().lower())
            if std is None:
                raise argparse.ArgumentTypeError(f"未知场景键: {kv!r}")
            v = v.strip()
            sc[std] = v if std == "label" else float(v)
        if sc:
            scenarios.append(sc)
    if not scenarios:
        raise argparse.ArgumentTypeError("--scenarios 解析为空")
    return scenarios


def _util_main_cli():
    """命令行入口: `python -m model_analyzer.util_sweep --scenarios ... --model ...`。

    parser 复用 model_analyzer() 签名 (所有 model/并行/透传参数自动覆盖),
    再外加 `--scenarios` (利用率网格) 和 `--output_xlsx`。
    """
    parser = build_cli_parser(
        model_analyzer,
        exclude=("_sweep_label", "util_tensor_core", "util_vector",
                 "util_sfu", "util_bandwidth", "generate_csv"),
        type_overrides=_MODEL_ANALYZER_CLI_TYPES,
        prog="python -m model_analyzer.util_sweep",
        description="利用率扫描: 对多组 {tensor_core,vector,sfu,bandwidth} 跑同一模型并合并对比")
    parser.add_argument(
        "--scenarios", required=True,
        help="利用率场景, 分号分隔; 每场景逗号分隔 key=val "
             "(键: tc/vec/sfu/bw/label). 例: 'tc=0.6,vec=0.3,sfu=0.3,bw=0.7;tc=0.8,bw=0.85'")
    parser.add_argument("--output_xlsx", default=None, help="合并表输出路径 (默认自动生成到 raw_data/)")
    args = parser.parse_args()
    kwargs = {k: v for k, v in vars(args).items() if v is not None}
    scenarios = _parse_scenarios(kwargs.pop("scenarios"))
    output_xlsx = kwargs.pop("output_xlsx", None)
    return model_analyzer_util_sweep(scenarios, output_xlsx=output_xlsx, **kwargs)


if __name__ == "__main__":
    _util_main_cli()
