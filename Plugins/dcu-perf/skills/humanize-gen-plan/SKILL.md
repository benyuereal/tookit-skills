---
name: humanize-gen-plan
description: Convert an LLM inference optimization request into a pre-registered experiment plan with immutable workload, control/candidate identity, correctness oracles, operator and E2E gates, environment fingerprints, rollback, artifact contract, milestones, and stop conditions. Use before remote GPU profiling, kernel work, serving configuration search, or multi-round optimization.
version: 1.0.0
author: DCU Engineer
license: MIT
---

# Humanize Gen Plan

## Overview

First phase of the Humanize/RLCR Loop Engineering workflow. Converts informal requirements and optimization goals into a structured, executable plan document that serves as the contract for subsequent automated optimization loops.

## When To Use

- Starting a new model performance optimization task
- Defining the scope and boundaries of an experiment
- Setting up a reproducible benchmarking and optimization loop
- Documenting what "SOTA" means in your specific context (fixed model, hardware, workload, SLA)
- Providing the input contract for `humanize-rlcr` automated loop execution

## Plan Structure

Generate a `plan.md` with these sections:

```markdown
# Performance Optimization Plan: <Task Name>

## 1. Target Description
- Model: <model name, path, precision, quantization>
- Hardware: <GPU type> × <count>, <driver/DTK version>
- Workload: <dataset, input/output distribution, request rate>
- Baseline: <current performance numbers>
- SLA Target: <latency/throughput requirements>

## 2. Acceptance Criteria
- [ ] Criterion 1: <measurable, specific>
- [ ] Criterion 2: <measurable, specific>
- [ ] Criterion 3: <measurable, specific>
- [ ] Candidate path activation proven by logs, maps, counters, or trace
- [ ] Same-window A/B exceeds the pre-registered noise/margin gate

## 3. Positive Tests (what must be true)
- Test 1: <condition that validates correctness>
- Test 2: <condition that validates correctness>

## 4. Negative Tests (what must NOT happen)
- Test 1: <condition that would invalidate the approach>
- Test 2: <condition that would invalidate the approach>

## 5. Path Boundaries
- In scope: <what the agent can modify>
- Out of scope: <what the agent must not touch>
- Constraint 1: <hard constraint>
- Constraint 2: <hard constraint>

## 5.1 Environment Fingerprint
- Framework/model/checkpoint/library hashes:
- Hardware, DTK/driver, container and co-tenant state:
- Required environment and fallback gates:

## 5.2 Decision and Recovery
- Operator gate:
- E2E gate and repetition count:
- Decision labels: ACCEPTED / REJECTED / INCONCLUSIVE / DIAGNOSTIC ONLY
- Exact rollback entrypoint:

## 6. Milestones
- [ ] M1: Fair benchmark search complete
- [ ] M2: Gap analysis and bottleneck identification
- [ ] M3: First optimization candidate tested
- [ ] M4: Best candidate meets SLA target
- [ ] M5: Final report and PR ready

## 7. Artifacts
- Benchmark results: <path>
- Profile traces: <path>
- Optimization patches: <path>
- Validation reports: <path>
```

## Six-Step Loop Engineering Flow (from Article)

1. **Define target boundary**: Same model, hardware, GPU count, precision, workload, SLA, framework commit, serving params
2. **Run fair benchmark search**: Use `llm-serving-auto-benchmark` to establish baseline
3. **Judge the gap**: Compare against target SLA and best-known configuration
4. **Profile to explain the gap**: Use `llm-torch-profiler-analysis` and `llm-pipeline-analysis`
5. **Patch only on evidence-backed paths**: Don't optimize blind
6. **Re-validate on same workload**: Every round records benchmark, profile, accuracy, failures

Pre-register one primary hypothesis per round. Include a physical/engineering feasibility bound and
stop when the remaining target requires platform or vendor changes outside the allowed scope.

## Example Plan

```markdown
# Performance Optimization Plan: Qwen3-8B TPOT Reduction

## 1. Target Description
- Model: Qwen/Qwen3-8B, FP16
- Hardware: 2× DCU (gfx936) via BW10 server
- Workload: ShareGPT dataset, Poisson arrival, 32 requests
- Baseline: TPOT 52ms, TTFT 180ms, Throughput 12.3 req/s
- SLA Target: TPOT < 35ms, TTFT < 150ms

## 2. Acceptance Criteria
- [ ] TPOT p50 < 35ms under target workload
- [ ] TTFT p50 < 150ms under target workload
- [ ] No accuracy regression (MMLU < 0.5% drop)
- [ ] Memory usage within 80% GPU budget

## 3. Positive Tests
- Benchmark repeatable within ±5% across 3 runs
- Profile trace shows expected kernel families
- CUDA graph active (no silent disable)

## 4. Negative Tests
- No NaN/Inf in attention outputs
- No OOM under sustained load
- No silent attention backend fallback

## 5. Path Boundaries
- In scope: serving parameters, attention config, torch.compile flags
- Out of scope: model architecture changes, custom kernel development
- Constraint: Must work with DTK 26.04 toolchain

## 6. Milestones
- [ ] M1: Fair benchmark baseline on target hardware
- [ ] M2: Profile analysis identifies TPOT bottleneck
- [ ] M3: Apply evidence-backed optimizations
- [ ] M4: Iterate until SLA met or evidence exhausted
- [ ] M5: Final report with reproducible config

## 7. Artifacts
- Benchmark results: benchmark_results/qwen3_8b_tpot/
- Profile traces: benchmark_results/qwen3_8b_tpot/profiles/
- Optimization patches: v0.23/patches/qwen3_tpot_opt.py
```

## Integration

- **Output**: Structured plan.md consumed by `humanize-rlcr`
- **Uses**: `llm-serving-auto-benchmark` for baseline establishment
- **Uses**: `llm-torch-profiler-analysis` for gap analysis planning
- **Uses**: `model-compute-simulation` for theoretical limit estimation
