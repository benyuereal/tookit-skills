---
name: model-pr-optimization-history
description: Use when an SGLang, vLLM, or TensorRT-LLM serving/model optimization task needs open-source community PR evidence. Query and read the PR-driven history docs under model-pr-optimization-history before choosing source paths, fast paths, kernel/fusion ideas, regression risks, or validation lanes.
version: 1.0.1
author: DCU Engineer
license: MIT
---

> ⚠ **来源标注**：本 skill 整包吸纳自 BBuf/AI-Infra-Auto-Driven-SKILLS v0.1.5（2026-10-05 tag，MIT License，https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS）。
> 上游方法论基于 CUDA 平台（SGLang/vLLM/TensorRT-LLM on NVIDIA）经验；**DCU/ROCm 上是否复现未经我们实测**，涉及具体性能/行为观察处按「参考、可尝试、DCU 待验证」对待。
> ⚠️ **数据未随本包分发**：`sglang/` `vllm/` `tensorrt_llm/` `tokenspeed/` 的 PR 历史数据（约 39.5MB，且含第三方 PR 内容）已从仓库剔除。如需使用请按需生成/下载；本包仅保留 SKILL.md + `scripts/query.py` 索引与查询接口。

# Model PR History Knowledge

This is the open-source community PR knowledge base for model optimization
history. It is not a set of per-model skills. Each model family keeps bilingual
docs with inspected PR diffs, implementation file coverage, timelines, changed
files, code excerpts, and validation/risk notes. Framework-wide PRs that cannot
be pinned to one model family should go into the framework `shared/` bucket
instead of being dropped.

Use it before patching model-specific serving paths, choosing an SGLang SOTA
optimization target, or explaining why a framework already has a faster path.
> Local fork evidence is not bundled in this package; consult the internal
> variant separately if available.

## Query

Run commands from this directory:

```bash
python3 scripts/query.py --list
python3 scripts/query.py --framework sglang --model qwen3-core --paths-only
python3 scripts/query.py --framework sglang --model qwen3-core "fused qk norm rope"
python3 scripts/query.py --framework vllm "DeepSeek-V4 fused norm router" --limit 5
```

Useful options:

- `--framework sglang|vllm|tensorrt_llm|tokenspeed`: restrict to one serving framework.
- `--model <slug>`: restrict to one model family directory.
- `--lang en|zh|both`: select English, Chinese, or both docs.
- `--paths-only`: print the exact docs to read without snippets.
- `--limit N`: bound search results.

## Workflow

1. Infer the model-family slug from the user's model id, checkpoint path, or
   SGLang source path. If unsure, run `scripts/query.py "<model name>"`.
2. Read the matching SGLang community PR history first for SGLang patch work. Read the vLLM
   history too when vLLM is the leading competitor or its trace suggests a
   missing SGLang fast path. If the trace points to a framework-wide runtime,
   parser, or utility gap, also check the framework `shared/` bucket. For local
fork evidence, this package no longer bundles the internal variant.
3. Extract only actionable evidence:
   - model implementation files and symbols
   - PRs that changed the hot source path
   - prior fusions, overlap work, quantization, MoE, attention, cache, sampler,
     or loader changes
   - open/watch PRs that may explain a known gap or pending support issue
   - validation lanes and regression risks implied by the PR cards
4. Save a short note in the active run artifacts, for example
   `history/model-pr-history-notes.md`, with paths read, PR numbers, source
   files, and the decision each item influenced.
5. Do not copy long PR cards into the final answer. Cite paths and summarize the
   relevant implementation/risk.

## Model Slugs

Current frameworks:

- `sglang`
- `vllm`

Each framework also exposes a `shared` bucket for relevant PRs that are real
but not safely attributable to one model family.

Current model-family slugs include:

```text
deepseek-v3-r1, deepseek-v31, deepseek-v32, deepseek-v4, ernie45,
gemma4, glm-vlm-ocr, glm45, glm46-glm47, glm5-glm51, gpt-oss,
hunyuan3-preview, intern-s1, internvl35, kimi, llama4, mimo-v2-flash,
minimax, mistral-small-4, mixtral-quark-int4fp8-moe, moss-vl,
nemotron-super, qwen-vlm-omni-asr, qwen3-coder, qwen3-core,
qwen3-next, qwen35, qwen36, step35
```

## SOTA Loop Contract

For `sglang-sota-humanize-loop`, this knowledge base is an early context
source:

- Read it after model identification and before patch planning.
- Include the history paths and key community PR evidence in `analysis/root-cause.md` or
  `history/model-pr-history-notes.md`.
- If the profiler points at a known model path, check whether the history has
  prior changes on that file before writing a new patch.
- If a competitor is faster, search that competitor's model history and the
local fork history is not bundled; search that competitor's model history
  before assuming the gap is kernel-local.
