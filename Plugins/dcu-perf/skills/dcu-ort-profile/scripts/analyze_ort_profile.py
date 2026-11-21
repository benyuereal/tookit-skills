#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ORT 算子耗时统计工具（DCU / ROCm / 通用）

解析 ONNX Runtime 通过 SessionOptions.enable_profiling = True
导出的 Chrome-Tracing 兼容 JSON 日志（onnxruntime_profile_*.json），
按「算子名 | 原始 Provider」为唯一键统计：
  - 调用次数、累计 dur(us)、单次平均 dur(us)
并输出终端表格 + 落盘 CSV。

适用：ORT 任意版本，跨平台（ROCm / CUDA / CPU EP 均可）。
来源：内部文档《ORT上的算子耗时统计工具》（file_id=<internal-file-id>）

用法：
    python3 analyze_ort_profile.py ort_profile.log            # 默认文件名
    python3 analyze_ort_profile.py onnxruntime_profile_*.json # 指定日志
输出：
    - 终端打印按累计 dur 降序的算子表
    - 同目录生成 ort_op_complete_analysis.csv
"""
import json
import csv
from collections import defaultdict


def format_provider(provider_str):
    """
    格式化 provider 名称为短设备名：
      - 缺失/空值 → 填充为 CPUExecutionProvider → 格式化后 CPU
      - ROCMExecutionProvider → ROCM
      - CUDAExecutionProvider → CUDA
      - CPUExecutionProvider  → CPU
      - 其他 → 保留原始前缀
    """
    if not provider_str or provider_str.strip() == "":
        provider_str = "CPUExecutionProvider"
    provider_str = provider_str.strip()
    if "ExecutionProvider" in provider_str:
        prefix = provider_str.split("ExecutionProvider")[0]
        return prefix if prefix else "Unknown"
    return provider_str  # 兼容无 ExecutionProvider 后缀的情况


def analyze_ort_profile_dur(log_file_path):
    """
    解析 ORT Profile 日志（首尾含 []），核心规则：
    1. 缺失 provider 时默认 CPUExecutionProvider（格式化后 CPU）
    2. 同一算子不同 provider 分开统计（算子名 + provider 作为唯一标识）
    """
    op_stats = defaultdict(lambda: {
        "op_name": "",
        "provider": "",
        "device": "",
        "count": 0,
        "total_dur": 0,
        "avg_dur": 0.0,
    })

    # 读取并预处理日志内容（处理首尾的 []）
    with open(log_file_path, "r", encoding="utf-8") as f:
        raw_content = f.read().strip()
        if raw_content.startswith("[") and raw_content.endswith("]"):
            clean_content = raw_content[1:-1].strip()
        else:
            clean_content = raw_content
        json_strs = [s.strip().rstrip(",") for s in clean_content.split("\n") if s.strip()]

    # 逐行解析 JSON 串
    for idx, json_str in enumerate(json_strs, 1):
        if not json_str:
            continue
        try:
            log_entry = json.loads(json_str)
            if log_entry.get("cat") != "Node":
                continue
            args = log_entry.get("args", {})
            op_name = args.get("op_name", f"unknown_op_{idx}")
            dur = log_entry.get("dur", 0)
            raw_provider = args.get("provider", "").strip()
            formatted_device = format_provider(raw_provider)
            final_provider = raw_provider if raw_provider.strip() else "CPUExecutionProvider"
            unique_key = f"{op_name}|{final_provider}"

            op_stats[unique_key]["op_name"] = op_name
            op_stats[unique_key]["provider"] = final_provider
            op_stats[unique_key]["device"] = formatted_device
            op_stats[unique_key]["count"] += 1
            op_stats[unique_key]["total_dur"] += dur
            op_stats[unique_key]["avg_dur"] = op_stats[unique_key]["total_dur"] / op_stats[unique_key]["count"]
        except json.JSONDecodeError as e:
            print(f"警告：第{idx}个 JSON 串解析失败: {e}，跳过该串")
            continue
        except Exception as e:
            print(f"警告：第{idx}个 JSON 串处理异常: {e}，跳过该串")
            continue

    # 按累计 dur 降序排序
    sorted_ops = sorted(op_stats.values(), key=lambda x: x["total_dur"], reverse=True)

    # 终端表格
    print("=" * 140)
    print(f"{'算子名':<25} {'运行设备':<10} {'原始Provider':<30} {'调用次数':<10} {'累计dur(微秒)':<15} {'单次平均dur(微秒)':<18} {'累计dur(毫秒)':<15}")
    print("-" * 140)
    for stats in sorted_ops:
        total_dur_ms = stats["total_dur"] / 1000
        print(
            f"{stats['op_name']:<25} {stats['device']:<10} {stats['provider']:<30} {stats['count']:<10} {stats['total_dur']:<15} {stats['avg_dur']:<18.2f} {total_dur_ms:<15.2f}"
        )
    print("=" * 140)

    # 统计汇总
    total_ops_type = len(sorted_ops)
    total_dur_all = sum([stats["total_dur"] for stats in sorted_ops])
    total_count_all = sum([stats["count"] for stats in sorted_ops])

    print(f"\n汇总统计：")
    print(f"- 算子+设备组合数：{total_ops_type} 种")
    print(f"- 总调用次数：{total_count_all:,} 次")
    print(f"- 总耗时：{total_dur_all:,} 微秒 ({total_dur_all/1000000:.4f} 秒)")

    device_stats = defaultdict(lambda: {"count": 0, "dur": 0})
    for stats in sorted_ops:
        device = stats["device"]
        device_stats[device]["count"] += stats["count"]
        device_stats[device]["dur"] += stats["total_dur"]

    print(f"\n按设备统计：")
    for device, stats in device_stats.items():
        print(f"- {device}：调用 {stats['count']:,} 次，耗时 {stats['dur']:,} 微秒 ({stats['dur']/1000000:.4f} 秒)")

    return sorted_ops


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        ORT_PROFILE_FILE = sys.argv[1]
    else:
        ORT_PROFILE_FILE = "ort_profile.log"

    sorted_result = analyze_ort_profile_dur(ORT_PROFILE_FILE)

    csv_path = "ort_op_complete_analysis.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "算子名", "运行设备", "原始Provider", "调用次数", "累计dur(微秒)",
            "单次平均dur(微秒)", "累计dur(毫秒)", "累计dur(秒)"
        ])
        for stats in sorted_result:
            total_dur = stats["total_dur"]
            writer.writerow([
                stats["op_name"],
                stats["device"],
                stats["provider"],
                stats["count"],
                total_dur,
                round(stats["avg_dur"], 2),
                round(total_dur / 1000, 2),
                round(total_dur / 1000000, 4),
            ])
    print(f"\n完整分析结果已保存到：{csv_path}")
