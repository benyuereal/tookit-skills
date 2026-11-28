#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""兼容 shim：迭代驱动的实现已迁到 `orchestrator.py`。

保留本文件是为了不打断既有命令（`python3 -I engine/driver.py ...`）与测试。
新代码请直接 import `orchestrator`。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from orchestrator import (  # noqa: E402,F401
    Backend, IterationRecord, FakeBackend, RealBackend,
    NOISE_FLOOR, MAX_NOOP, MAX_PLATEAU,
    agent_changes, correctness_of_decision, decide_verdict, noise_floor_from,
    render_student_prompt, iterate, repo_records_dir, main,
)

if __name__ == "__main__":
    raise SystemExit(main())
