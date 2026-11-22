---
name: llm-serving-capacity-planner
description: "从启动日志、模型几何、checkpoint 字节、KV cache 格式、graph pool、临时 workspace、allocator 行为、TP/PP/EP 拓扑来规划与审计显存容量（capacity planner）。何时用：fit check、OOM 或 VMFault 诊断、并发上限、mem-fraction 调参、量化 checkpoint 评估。"
version: 1.0.0
author: DCU Engineer
license: MIT
kind: tool
---

# LLM Serving Capacity Planner

## Overview

Parse and analyze LLM serving framework startup logs to understand capacity constraints. Produces a structured capacity report covering:
- Weight memory allocation
- KV cache budget and utilization
- CUDA graph memory overhead
- Maximum concurrent requests under current config
- OOM pressure points and headroom

## When To Use

- Deploying a new model and need to estimate capacity
- Getting OOM errors during serving startup or under load
- Tuning `--gpu-memory-utilization` or `--max-num-seqs`
- Comparing memory efficiency across frameworks (SGLang vs vLLM)
- Planning multi-model or multi-instance deployments on shared GPU

## Workflow

### Evidence gate

Collect the actual checkpoint tensor bytes or safetensors headers when quantization, experts or
auxiliary draft weights make parameter-count estimates inaccurate. Record per-rank residency after
warmup with the platform monitor and reconcile it with startup logs. Do not treat free memory before
graph capture as serving headroom.

Account separately for:

- weights, scales, embeddings and draft/speculative modules;
- KV cache by layer/head/dtype/block and target context/concurrency;
- graph-private pools and captured shapes;
- attention/MoE/collective workspaces and persistent output buffers;
- allocator fragmentation, lazy allocation and temporary dequantized slabs;
- duplicated rank/process state and host-pinned memory.

Fail closed when the exact path or dtype is unknown. Historical runs showed that configuration
labels can differ from the live quantization path, and that a memory-saving candidate may still OOM
only after graph capture or multi-rank warmup.

### 1. Collect Startup Logs

Capture the full server startup output:

```bash
# SGLang
python -m sglang.launch_server --model <model> --tp 2 2>&1 | tee startup.log

# vLLM
vllm serve <model> --tensor-parallel-size 2 2>&1 | tee startup.log
```

### 2. Parse Key Capacity Metrics

Extract and analyze:

| Metric | Where to Find | What It Tells You |
|--------|--------------|-------------------|
| **Weight Memory** | Startup log: "Loading model weights" | How much GPU memory the model parameters consume |
| **KV Cache Budget** | "--max-num-seqs" × per-seq KV size | How many concurrent requests fit |
| **CUDA Graph Overhead** | "Capturing CUDA graphs" section | Memory reserved for graph replays |
| **Peak Memory** | "GPU memory usage" or nvidia-smi/hy-smi | Total GPU memory consumed |
| **Headroom** | GPU total - peak memory | Memory available for batching/overhead |

### 3. Calculate Max Concurrent Requests

```
max_concurrent = kv_cache_budget / kv_cache_per_request

where:
  kv_cache_per_request = num_layers × 2 × num_kv_heads × head_dim × max_seq_len × dtype_bytes × tp_factor
```

### 4. Identify OOM Pressure Points

Common OOM scenarios and their signatures:

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| OOM at startup | Model too large for GPU(s) | Reduce TP, use quantization, split across more GPUs |
| OOM during warmup | CUDA graph capture too large | Reduce `--max-num-seqs`, disable piecewise CG |
| OOM under load | KV cache exhausted | Reduce `--max-num-seqs` or `--max-model-len` |
| Gradual OOM | Memory fragmentation or leak | Check for growing caches, restart periodically |

### 5. Generate Capacity Report

Output a structured report:

```markdown
## Capacity Report: <model> on <GPU> × <count>

### Memory Breakdown
- Total GPU Memory: X GiB
- Model Weights: Y GiB (Z% of total)
- KV Cache Budget: W GiB (V% of total)
- CUDA Graph Overhead: U GiB
- Other (activations, workspace): T GiB
- Free Headroom: S GiB

### Capacity
- Max Concurrent Requests: N
- Max Supported Context Length: L tokens
- Effective Batch Size at max_seq: B

### Recommendations
- [Actionable recommendations based on findings]
```

## Memory Estimation Formulas

### Weight Memory
```
weight_memory = total_params × dtype_bytes
# Example: 8B FP16 model = 8B × 2 = 16 GiB
# With TP=2: 16 / 2 = 8 GiB per GPU
```

### KV Cache per Token
```
kv_cache_per_token = 2 × num_layers × num_kv_heads × head_dim × dtype_bytes
# Example: 32 layers, 8 KV heads, 128 dim, FP16
# = 2 × 32 × 8 × 128 × 2 = 131,072 bytes ≈ 128 KB/token
```

### CUDA Graph Memory (approximate)
```
cuda_graph_memory ≈ num_graphs × graph_replay_size
# Typically 1-5 GiB depending on max_num_seqs and model size
```

## Integration with Other Skills

Validate the resulting concurrency with `llm-serving-auto-benchmark`. Re-check capacity after each
accepted kernel or graph change; persistent buffers and captured graphs alter the budget. For a
VMFault with apparent memory headroom, route to `debug-cuda-crash` instead of assuming ordinary OOM.

- Before benchmarking: use this to verify the model fits on target hardware
- After OOM: use this to identify which memory pool is exhausted
- With `llm-serving-auto-benchmark`: feed capacity constraints into search space bounds
- With `sglang-prod-incident-triage`: distinguish OOM from other failure modes
