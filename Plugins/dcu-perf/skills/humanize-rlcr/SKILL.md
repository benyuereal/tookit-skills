---
name: humanize-rlcr
description: "Run a resumable evidence-driven LLM inference (serving/整模型) optimization loop from a structured plan produced by humanize-gen-plan: repeated baseline, profile, patch, correctness, same-window A/B, acceptance, rejection, rollback and artifact-review rounds on unstable remote GPU environments. 触发场景：手上已有 humanize-gen-plan 生成的计划、要按轮次推进并校验每轮证据。不适用：单个算子 kernel 的优化循环（candidates.yaml + loop.py + 候选队列）走 ops 域 op-optimize；DCU 方法论总纲走 dcu-op-optimization。"
version: 1.0.1
author: DCU Engineer
license: MIT
kind: tool
---

# Run an Evidence-Driven Optimization Loop

Use a plan from `humanize-gen-plan`. Keep one primary hypothesis per round and fail closed.

## Round workflow

1. Restore the last accepted state and verify hashes, maps, service health and correctness.
2. Establish a same-window control baseline and environment fingerprint.
3. Profile or inspect source only enough to test the registered hypothesis.
4. Implement the smallest gated candidate with a documented rollback.
5. Pass operator, graph/eager, E2E token, concurrency and stability correctness gates as needed.
6. Run interleaved control/candidate A/B under the exact registered workload.
7. Decide `ACCEPTED`, `REJECTED`, `INCONCLUSIVE`, or `DIAGNOSTIC ONLY`.
8. Restore the accepted service after a rejected or inconclusive candidate.
9. Archive evidence and register the next single hypothesis.

Never stack unaccepted candidates. Do not continue merely because a standalone kernel is faster.

## Artifact contract

```text
round_N/
├── PLAN.md
├── REPORT.md
├── manifest.json
├── results/
│   ├── correctness.json
│   └── benchmark.json
├── artifacts/
└── rollback/
```

`manifest.json` must contain `workload`, `environment`, `control`, `candidate`, and `artifacts`.
Include paths or hashes, not credentials, models, binaries, large traces or raw production logs.

Validate a completed round:

```bash
SKILL_DIR=/path/to/humanize-rlcr
python3 "$SKILL_DIR/scripts/validate_round.py" /path/to/round_N
```

## Review gates

- **State**: prove the actual candidate path ran and identify the prior accepted anchor.
- **Correctness**: require the strongest appropriate oracle, not just readable text or health 200.
- **Evidence**: account for workload equality, profiler inflation and host non-stationarity.
- **Performance**: require repeated same-window improvement above the registered threshold.
- **Risk**: check multi-rank, concurrency, graph, quantization, memory and fallback behavior.
- **Recovery**: prove the exact PID/container/config rollback before accepting.

Stop when the target is met, the physical/engineering bound makes it infeasible, evidence-backed
paths are exhausted, or platform changes are required. Record infeasibility as a result rather
than manufacturing more optimization rounds.
