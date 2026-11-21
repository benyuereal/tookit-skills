#!/usr/bin/env python3
"""Summarize Chrome traces by kernel family and explicit layer ranges."""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


LAYER_RE = re.compile(r"(?:model[._/])?layers?[._/\[](?P<layer>\d+)", re.I)


def load_trace(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        payload = json.load(handle)
    events = payload.get("traceEvents", payload) if isinstance(payload, dict) else payload
    if not isinstance(events, list):
        raise ValueError("trace must contain a traceEvents list")
    return [event for event in events if isinstance(event, dict)]


def complete(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event for event in events if event.get("ph") == "X" and event.get("dur", 0) > 0]


def is_gpu(event: dict[str, Any]) -> bool:
    cat = str(event.get("cat", "")).lower()
    args = event.get("args", {})
    return (
        any(token in cat for token in ("kernel", "gpu", "cuda", "hip"))
        or (isinstance(args, dict) and any(key in args for key in ("stream", "device", "External id")))
    )


def family(name: str) -> str:
    lowered = name.lower()
    groups = (
        ("collective", ("nccl", "rccl", "allreduce", "all_reduce")),
        ("attention", ("attention", "flash", "paged", "mqa", "mla")),
        ("moe", ("moe", "expert", "topk", "router")),
        ("gemm", ("gemm", "gemv", "rocblas", "hipblas", "mm_")),
        ("norm", ("norm", "rms")),
        ("quant", ("quant", "dequant", "fp4", "fp8", "w8a8")),
        ("elementwise", ("elementwise", "silu", "gelu", "hadamard", "reduce")),
        ("copy", ("memcpy", "copy", "clone")),
    )
    for label, tokens in groups:
        if any(token in lowered for token in tokens):
            return label
    return "other"


def print_table(rows: list[tuple[str, float, int]], total: float) -> None:
    print("| Group | Duration ms | Share | Calls |")
    print("|---|---:|---:|---:|")
    for label, duration, calls in rows:
        share = 100.0 * duration / total if total else 0.0
        print(f"| {label} | {duration / 1000.0:.3f} | {share:.2f}% | {calls} |")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--config", type=Path, help="Optional model config; recorded for provenance")
    args = parser.parse_args()

    events = complete(load_trace(args.trace))
    gpu_events = [event for event in events if is_gpu(event)]
    totals: dict[str, list[float]] = defaultdict(list)
    for event in gpu_events:
        totals[family(str(event.get("name", "unknown")))].append(float(event["dur"]))

    total_duration = sum(sum(values) for values in totals.values())
    rows = sorted(
        ((label, sum(values), len(values)) for label, values in totals.items()),
        key=lambda row: row[1],
        reverse=True,
    )
    print("# Pipeline Trace Summary")
    print(f"\nTrace: `{args.trace}`")
    if args.config:
        print(f"Config: `{args.config}`")
    print("\n## Kernel families\n")
    print_table(rows, total_duration)

    layer_ranges: list[tuple[int, float, float]] = []
    for event in events:
        match = LAYER_RE.search(str(event.get("name", "")))
        if match:
            start = float(event.get("ts", 0))
            layer_ranges.append((int(match.group("layer")), start, start + float(event["dur"])))

    print("\n## Explicit layer ranges\n")
    if not layer_ranges:
        print("No explicit layer ranges were present. Do not infer per-layer attribution from kernel names alone.")
        return 0

    layer_totals: dict[int, float] = defaultdict(float)
    for layer, start, end in layer_ranges:
        for event in gpu_events:
            timestamp = float(event.get("ts", 0))
            if start <= timestamp < end:
                layer_totals[layer] += float(event["dur"])
    layer_rows = [(f"layer {layer}", duration, 1) for layer, duration in sorted(layer_totals.items())]
    print_table(layer_rows, sum(layer_totals.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
