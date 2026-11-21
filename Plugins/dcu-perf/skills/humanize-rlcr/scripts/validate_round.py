#!/usr/bin/env python3
"""Validate the minimum evidence contract for one optimization round."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_REPORT_DECISIONS = (
    "ACCEPTED",
    "REJECTED",
    "INCONCLUSIVE",
    "DIAGNOSTIC ONLY",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("round_dir", type=Path)
    args = parser.parse_args()
    root = args.round_dir

    failures: list[str] = []
    for name in ("PLAN.md", "REPORT.md", "manifest.json"):
        if not (root / name).is_file():
            failures.append(f"missing {name}")

    for name in ("correctness.json", "benchmark.json"):
        path = root / "results" / name
        if not path.is_file():
            failures.append(f"missing results/{name}")
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            failures.append(f"invalid results/{name}: {exc}")

    manifest = root / "manifest.json"
    if manifest.is_file():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            for key in ("workload", "environment", "control", "candidate", "artifacts"):
                if key not in data:
                    failures.append(f"manifest missing key: {key}")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            failures.append(f"invalid manifest.json: {exc}")

    report = root / "REPORT.md"
    if report.is_file():
        text = report.read_text(encoding="utf-8", errors="replace")
        if not any(decision in text for decision in REQUIRED_REPORT_DECISIONS):
            failures.append("REPORT.md has no explicit decision")
        for phrase in ("correctness", "benchmark", "rollback"):
            if phrase.lower() not in text.lower():
                failures.append(f"REPORT.md does not mention {phrase}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print(f"PASS: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
