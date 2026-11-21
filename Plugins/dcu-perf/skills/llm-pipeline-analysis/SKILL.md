---
name: llm-pipeline-analysis
description: Decompose LLM serving traces into mutually exclusive latency groups, explicit layer ranges, kernel families, streams, and rank tails. Use after global torch-profiler triage to localize prefill, decode, MoE, attention, GEMV, collective, elementwise, scheduler, or speculative-decoding bottlenecks without overclaiming attribution hidden by graph capture.
version: 1.0.0
author: DCU Engineer
license: MIT
---

# Analyze an LLM Pipeline Trace

Build an additive latency budget that can guide one evidence-backed optimization hypothesis.

## Gather evidence

Require:

- A Chrome trace from the production workload and its server/workload manifest.
- A model `config.json` when layer geometry matters.
- Wall-clock TTFT/TPOT and exact output-token counts from the same run.
- A graph-off mapping trace when the graph-on trace hides Python or layer ranges.

Run the bundled conservative summarizer:

```bash
SKILL_DIR=/path/to/llm-pipeline-analysis
python3 "$SKILL_DIR/scripts/analyze_llm_pipeline.py" \
  --trace /path/to/trace.json.gz \
  --config /path/to/model/config.json
```

The script reports kernel families and attributes layers only when explicit layer ranges exist.
It deliberately refuses to infer layer identity from kernel names alone.

## Build mutually exclusive budgets

Separate prefill, decode, draft and verify. Within each stage, account for:

- attention;
- dense GEMM/GEMV;
- routed/shared MoE;
- collective communication;
- quant/dequant;
- norm, elementwise and reduction;
- copy/allocation;
- scheduler, Python and other glue.

Kernel sums can exceed HTTP wall time because of overlap and profiler inflation. Reconcile the
difference explicitly; do not normalize it away.

## Decide the next experiment

For each group, record duration, calls per token/cycle, source path, production shape, physical
floor, current implementation and closure evidence. Project candidate operator savings to E2E:

```text
projected_E2E = operator_delta * steady_state_calls - integration_cost
```

Reject hypotheses below the pre-registered noise/margin gate. Historical DCU/HCU runs show that
Python wrappers, allocation, layout conversion and rank desynchronization can erase more than a
5x standalone kernel gain.

Re-profile after every accepted change because the dominant group moves. Return one primary
hypothesis, its projected upper bound, required correctness oracle and E2E acceptance threshold.
