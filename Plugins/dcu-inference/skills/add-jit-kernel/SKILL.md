---
name: add-jit-kernel
description: "给上游 SGLang 加轻量 JIT CUDA kernel（sglang jit kernel）：bindings、正确性测试、基准、CI 注册、服务路径验证。何时用：checkout 里有 python/sglang/jit_kernel 且算子适配其 CUDA 工具链；平台原生实现走别的流程。"
version: 1.0.0
author: DCU Engineer
license: MIT
kind: tool
---

# Add a JIT Kernel to SGLang

Use this workflow for a lightweight operator in upstream SGLang's `python/sglang/jit_kernel` module. Read [the detailed upstream tutorial](references/upstream-jit-kernel-tutorial.md) for current file templates, utilities, registration examples, tests, and benchmark scaffolding.

## 1. Pass the Scope Gate

Before editing, verify all of the following:

- The current checkout contains `python/sglang/jit_kernel` and its local instructions have been read.
- The target is supported by this module's CUDA JIT toolchain.
- The operator does not require a heavyweight external project such as CUTLASS; otherwise consider `add-sgl-kernel`.
- Existing PyTorch, Triton, vendor, or JIT operators cannot satisfy the production shapes.

This skill is not the integration path for independent HIP/GCN shared libraries loaded through ctypes. Do not translate CUDA intrinsics, CI labels, or binary assumptions to DCU/HCU gfx928, gfx936, or gfx938 without an explicit port and target-hardware validation.

## 2. Register the Experiment Contract

Write down one hypothesis and freeze:

- commit, model, hardware, driver/toolkit, dtype, layout, and graph mode;
- production shapes, call count per token or forward, batch and concurrency;
- control implementation and candidate implementation;
- numerical tolerances and exact output checks;
- benchmark metric, sample count, noise threshold, and rollback.

Estimate the end-to-end ceiling before implementation:

```text
projected_gain_ms = operator_delta_ms * production_call_count
net_gain_ms = projected_gain_ms - integration_cost_ms
```

Integration cost includes binding, dispatch, allocation, layout conversion, compilation, graph breaks, synchronization, and downstream collective effects.

## 3. Implement the Smallest Complete Operator

Follow the detailed tutorial and current neighboring kernels for:

1. CUDA source and supported architecture guards.
2. C++ binding and validation of device, dtype, shape, stride, and alignment.
3. Python wrapper with an explicit fallback and stable public API.
4. Build/source registration in the current JIT module.
5. A path-activation mechanism such as a debug counter, one-time load log, or profiler range.

Do not add hidden allocations or device synchronization to a hot wrapper. Reuse output/workspace storage where the contract allows it. Compile failures and unsupported shapes must fail clearly or take the documented fallback.

## 4. Run the Correctness Ladder

Validate in this order:

1. The expected source compiled for the intended architecture and the expected artifact loaded.
2. Production shapes, dtypes, layouts, strides, boundary sizes, and misalignment cases.
3. NaN/Inf and the operator's exact numerical semantics against a trusted reference.
4. Eager and graph-mode agreement where graph execution is supported.
5. End-to-end token IDs or output hashes with exact output-token counts.
6. Single-request and target-concurrency execution to expose cache, workspace, and stream reuse bugs.
7. A soak run long enough to cross allocator and cache reuse boundaries.

Cosine similarity alone is insufficient when individual element errors or token divergence matter. Correct output also does not prove that the candidate ran; retain path-activation evidence.

## 5. Add Tests and Benchmarks

Use the current checkout's `test/README.md` and `write-sglang-test` skill. Keep correctness tests and timing benchmarks separate.

The operator benchmark must include reference and candidate under identical input preparation, separated warmup, production shapes, median and distribution, correctness before timing, and environment metadata.

Run the serving benchmark as interleaved control/candidate samples in the same clean host window. Capture active GPU processes, health, memory, clocks, and host contention. Reject a result below the pre-registered noise threshold.

For autoregressive serving, define TPOT consistently:

```text
TPOT = (total_time_ms - TTFT_ms) / (output_tokens - 1)
```

## 6. Decide and Integrate

Accept only when the candidate passes correctness and activation gates, improves the registered metric beyond noise, does not regress production shapes or concurrency, and has an appropriate fallback or kill switch.

If rejected or inconclusive, restore the last accepted state and archive the hypothesis, evidence, and reason. Do not stack another optimization on an unaccepted candidate.

## Deliverables

- kernel, binding, Python API, and fallback;
- activation evidence and correctness tests;
- CI registration and benchmark artifacts;
- environment manifest and exact commands;
- accept/reject/inconclusive decision with rollback.
