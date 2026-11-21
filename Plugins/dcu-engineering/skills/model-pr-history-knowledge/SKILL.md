---
name: model-pr-history-knowledge
description: Build and query an evidence-backed history of model and runtime decisions, including accepted, rejected, inconclusive, and reverted optimization attempts. Use during SGLang code review, regression analysis, architecture evolution, experiment planning, or when a previous rationale and its hardware/workload scope must be recovered.
version: 1.0.0
author: DCU Engineer
license: MIT
---

# Model PR History Knowledge

## Overview

Maintain a living archive of PR-driven model evolution decisions. This skill captures **why** specific implementation choices were made, creating a searchable knowledge base that stays close to the source code it documents. Paired with `sglang-humanize-review` to provide historical context during code review.

## Core Principle

> Archive the WHY, not just the WHAT.

When a PR changes model code (kernel, pipeline, config, scheduler), capture:
- **Decision**: What approach was chosen
- **Alternatives**: What other approaches were considered and rejected
- **Rationale**: Why this approach was chosen (performance, correctness, maintainability)
- **Evidence**: Benchmarks, test results, discussion links
- **Risks**: Known limitations or future work items
- **Decision status**: accepted, rejected, inconclusive, reverted, or superseded
- **Scope**: exact hardware, toolkit, model, shape, concurrency, graph mode, and commit
- **Recovery**: rollback commit or instructions that restore the last accepted state

## When To Use

- Reviewing a PR and need to understand why previous decisions were made
- Investigating a regression and need to trace back to the change that introduced it
- Proposing a new change and want to avoid repeating past mistakes
- Onboarding new contributors who need to understand model evolution
- Performing code archaeology on a complex model (e.g., why is attention implemented this way?)
- Verifying that a new PR doesn't contradict an earlier deliberate decision

## Data Model

### PR History Entry

```yaml
pr_number: "#12345"
title: "Optimize flash attention for H200 with FP8"
date: "2025-06-15"
author: "maintainer-name"
files_changed:
  - path: "python/sglang/srt/layers/attention/flash_attention.py"
    rationale: "Added FP8 fast path for H200 Hopper+ GPUs"
  - path: "python/sglang/srt/model_executor/models/qwen3.py"
    rationale: "Enabled FP8 attention dispatch for Qwen3 model config"
decision:
  summary: "Use FP8 flash attention for H200 to reduce memory bandwidth by 2x"
  alternatives_considered:
    - "INT8 quantization: rejected because precision loss too high for attention"
    - "Sparse attention: rejected because sparsity pattern varies by model"
  rationale: "H200 FP8 delivers 2x throughput with <0.1% perplexity regression"
evidence:
  benchmarks: "benchmark_results/fp8_attn_h200_20250615.json"
  discussion: "https://github.com/sgl-project/sglang/pull/12345"
risks:
  - "Not tested on A100 (no FP8 support)"
  - "May need fallback path for dynamic shapes"
```

### Model Evolution Timeline

```markdown
## Qwen3-8B Attention Evolution

| PR | Date | Change | Rationale |
|----|------|--------|-----------|
| #12001 | 2025-05-01 | Baseline: FlashInfer backend | Default for all models |
| #12345 | 2025-06-15 | Add FP8 attention for H200 | 2x throughput on Hopper |
| #12456 | 2025-07-01 | Fix FP8 numerical stability | NaN with long context |
| #12567 | 2025-07-10 | Add sliding window support | Qwen3 specific feature |
```

## Workflow

### 1. Extract PR Context

When a new PR is submitted, extract the decision context:

```bash
# From PR description and discussion
# Key sections to parse:
# - Motivation / Problem statement
# - Approach / Design decisions
# - Alternatives considered
# - Benchmark results
# - Known limitations
```

### 2. Categorize by Decision Type

| Decision Type | Example | Archive Location |
|--------------|---------|-----------------|
| **Kernel selection** | "Use flashinfer instead of triton" | `kernel-decisions.yaml` |
| **Architecture change** | "Split pipeline into 3 stages" | `architecture-decisions.yaml` |
| **Config tuning** | "Set default TP to 2 for 70B" | `config-decisions.yaml` |
| **Numerical precision** | "Use BF16 instead of FP16 for QK" | `precision-decisions.yaml` |
| **Dependency choice** | "Upgrade torch to 2.5 for compile" | `dependency-decisions.yaml` |

### 3. Link to Source Code

Create lightweight annotations that link decisions to the code:

```python
# In python/sglang/srt/layers/attention/flash_attention.py
#
# PR-HISTORY:
# - PR #12345: Added FP8 fast path for H200 (2025-06-15)
#   Rationale: 2x throughput on Hopper GPUs
# - PR #12456: Fixed numerical stability for long context (2025-07-01)
#   Rationale: NaN when seq_len > 32768 with FP8
def flash_attention_forward(...):
    ...
```

### 4. Query History During Review

When reviewing a PR, query the history:

```
Questions to answer from history:
1. Has this file been changed for similar reasons before?
2. Were there rejected alternatives that are relevant now?
3. Are there known risks from past changes that apply here?
4. Does this change contradict an earlier deliberate decision?
```

### 5. Maintain the Archive

Update the archive as part of the round or PR closeout. This skill does not bundle an automatic history writer: derive entries from Git, plans, reports, benchmark artifacts, profiler traces, and review discussion, then verify every link and measurement. Never invent missing rationale. Mark uncertainty explicitly and preserve negative evidence so later work does not repeat a rejected experiment.

## Archive Structure

```
.qoder/pr-history/
├── models/
│   ├── qwen3-8b/
│   │   ├── timeline.md          # Chronological change timeline
│   │   ├── attention.md         # Attention-related decisions
│   │   ├── memory.md            # Memory/quantization decisions
│   │   └── config.md            # Config/serving decisions
│   └── deepseek-v3/
│       └── ...
├── kernel-decisions.yaml        # Cross-model kernel decisions
├── architecture-decisions.yaml  # Cross-model architecture decisions
└── dependency-decisions.yaml    # Dependency upgrade decisions
```

## Integration with sglang-humanize-review

This skill works as the **context provider** for the review process:

```
Review Flow:
1. model-pr-history-knowledge → Provide historical context
2. sglang-humanize-review → Apply maintainer review standards
3. model-pr-history-knowledge → Archive new decision after merge
```

When reviewing, the agent should:
1. Query PR history for each file changed in the new PR
2. Check if the change aligns with or contradicts past decisions
3. If contradicting, flag it and request explicit rationale
4. After merge, archive the new decision with full context

## Output Contract

After a review session, produce:

```markdown
## PR History Context: PR #12567

### Relevant Past Decisions
- PR #12345 (2025-06-15): FP8 attention added → this PR builds on it
- PR #12001 (2025-05-01): FlashInfer baseline → still the fallback

### Consistency Check
- ✅ Aligns with: FP8 optimization path (PR #12345)
- ⚠️ Potential conflict: Assumes H200 FP8 availability (not portable)
- ❌ Contradicts: None found

### New Decision to Archive
- Decision: Add sliding window support to FP8 attention
- Rationale: Qwen3 uses sliding window for long context efficiency
- Risk: Interaction with FP8 precision at window boundaries

### Files to Annotate
- [ ] `flash_attention.py` — add PR-HISTORY annotation
- [ ] `qwen3.py` — add PR-HISTORY annotation
```

## Integration

- **Paired with**: `sglang-humanize-review` — review + history = complete quality gate
- **Feeds into**: `humanize-rlcr` — Loop Engineering uses history to avoid repeating mistakes
- **Consumed by**: All skills — any agent making PR changes should check history first
- **Reference**: SGLang PR conventions and maintainer discussion patterns
