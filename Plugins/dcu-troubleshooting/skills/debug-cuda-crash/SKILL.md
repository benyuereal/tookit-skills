---
name: debug-cuda-crash
description: "调试 SGLang / vLLM 里的 CUDA 或 HIP kernel 崩溃（debug cuda crash）：调用前日志、同步复现、张量校验、架构感知工具。何时用：非法内存访问、device 断言、VMFault、NaN/Inf、OOM，或怀疑自定义算子写坏，跨 NVIDIA/DCU/HCU。"
version: 1.0.0
author: DCU Engineer
license: MIT
kind: knowledge
---

# Tutorial: Debugging CUDA Crashes with Kernel API Logging

## Platform and Safety Gate

First record the exact device architecture, driver/toolkit, framework build, rank, process ID, and loaded shared-object path. A binary built for gfx928, gfx936, gfx938, or another target is not portable. On HIP/DCU/HCU, translate CUDA-only tools to the locally available HIP/DTK equivalents; do not present `compute-sanitizer`, `cuda-gdb`, or NCCL-specific output as evidence when those tools are unavailable.

Before adding logging, reproduce once with the smallest production-representative shape and synchronization enabled. Preserve the first failure, kernel/API log, `dmesg` or VMFault evidence, and per-rank status. Never reset a device, kill unrelated processes, or clear a shared server without explicit authorization.

This tutorial shows how to debug CUDA/HIP crashes and errors in SGLang (and similar frameworks like vLLM) using the `@debug_kernel_api` logging decorator.

## Goal

When your code crashes with CUDA errors such as illegal memory access, device-side assert, out-of-bounds, or NaN/Inf, use kernel API logging to:
- Capture input tensors BEFORE the crash occurs
- Understand what data caused the problem
- Track tensor shapes, dtypes, and values through the call boundary that triggered the crash
- Detect numerical issues such as NaN, Inf, or obviously wrong shapes

## Why Use Kernel API Logging?

**Problem**: CUDA/HIP errors often crash the program before normal debugging output is flushed.

**Solution**: SGLang's `@debug_kernel_api` decorator logs inputs before execution, so you can still see what caused the crash even after the program aborts.

## What Is Covered?

The current logging coverage focuses on the highest-value kernel boundaries in SGLang:
- Custom ops registered through `register_custom_op(...)`
- External custom ops registered through `register_custom_op_from_extern(...)`
- LLM attention, linear, quantization, and multi-platform wrapper entry points
- Diffusion attention impl, linear, rotary, and custom-op wrapper entry points
- Selected direct `torch.ops.sglang.*` hotspots and model-specific bypasses

This means the logging is useful for both LLM and diffusion kernel debugging, but it does not automatically cover every pure PyTorch call in the repository.

## Step 1: Enable Kernel API Logging

### Basic Logging (Function Names Only)

```bash
export SGLANG_KERNEL_API_LOGLEVEL=1
export SGLANG_KERNEL_API_LOGDEST=stdout

python my_script.py
```

Output:
```
================================================================================
[2026-03-19 00:47:06] SGLang Kernel API Call: RMSNorm.forward
================================================================================
[2026-03-19 00:47:06] SGLang Kernel API Call: sglang.quant_method.UnquantizedLinearMethod.apply
================================================================================
[2026-03-19 00:47:06] SGLang Kernel API Call: sglang.custom_op.fused_inplace_qknorm
```

### Detailed Logging (Inputs with Metadata)

```bash
export SGLANG_KERNEL_API_LOGLEVEL=3
export SGLANG_KERNEL_API_LOGDEST=debug.log

python my_script.py
```

### Full Logging (With Tensor Statistics)

```bash
export SGLANG_KERNEL_API_LOGLEVEL=5
export SGLANG_KERNEL_API_LOGDEST=debug.log

python my_script.py
```

### Crash-Safe Dumps (Inputs Saved Before Execution)

```bash
export SGLANG_KERNEL_API_LOGLEVEL=10
export SGLANG_KERNEL_API_LOGDEST=debug.log
export SGLANG_KERNEL_API_DUMP_DIR=/tmp/sglang_kernel_api_dumps

python my_script.py
```

At level 10, SGLang saves the inputs before execution. If the kernel crashes, the dump directory still contains the inputs and exception metadata.

If CUDA graph capture is active, tensor dumps are skipped automatically to avoid capture-time CUDA errors. In that case, you still get the kernel API call log, but not `inputs.pt` / `outputs.pt`.

## Step 2: Reproduce an LLM CUDA Crash

```bash
python3 - <<'PY'
from pathlib import Path
Path("/tmp/sglang_llm_crash.py").write_text(
    "import torch\n"
    "import torch.nn.functional as F\n"
    "from sglang.srt.utils.custom_op import register_custom_op\n\n"
    "def _fake_embedding(indices, table):\n"
    "    return torch.empty((*indices.shape, table.shape[-1]), device=table.device, dtype=table.dtype)\n\n"
    "@register_custom_op(op_name='mock_llm_cuda_crash', fake_impl=_fake_embedding)\n"
    "def mock_llm_cuda_crash(indices, table):\n"
    "    out = F.embedding(indices, table)\n"
    "    torch.cuda.synchronize()\n"
    "    return out\n\n"
    "table = torch.randn(4, 8, device='cuda', dtype=torch.float16)\n"
    "indices = torch.tensor([0, 7], device='cuda', dtype=torch.long)\n"
    "mock_llm_cuda_crash(indices, table)\n"
)
PY

SGLANG_KERNEL_API_LOGLEVEL=1 \
SGLANG_KERNEL_API_LOGDEST=/tmp/sglang_llm_level1.log \
python3 /tmp/sglang_llm_crash.py
```

## Step 3: Multi-Process Debugging

When running with multiple GPUs or worker processes, use `%i` in the log path:

```bash
export SGLANG_KERNEL_API_LOGLEVEL=3
export SGLANG_KERNEL_API_LOGDEST=debug_rank_%i.log

torchrun --nproc_per_node=4 my_script.py
```

## Step 4: Filter Level-10 Dumps

```bash
export SGLANG_KERNEL_API_LOGLEVEL=10
export SGLANG_KERNEL_API_LOGDEST=debug.log
export SGLANG_KERNEL_API_DUMP_DIR=/tmp/sglang_kernel_api_dumps
export SGLANG_KERNEL_API_DUMP_INCLUDE='sglang.custom_op.*'
export SGLANG_KERNEL_API_DUMP_EXCLUDE='*.fake_impl'
```

## Step 5: Common CUDA Errors and What to Check

### Illegal Memory Access or Device-Side Assert

Use: `export SGLANG_KERNEL_API_LOGLEVEL=3`

Check in the logs: tensor shapes, dtypes, CUDA vs CPU device placement, tensor stride / contiguity.

### NaN or Inf

Use: `export SGLANG_KERNEL_API_LOGLEVEL=5`

Check: `min`, `max`, `mean`, `nan_count`, `inf_count`.

### Out of Memory

Use: `export SGLANG_KERNEL_API_LOGLEVEL=3`

Check: unexpectedly large tensor shapes, batch size, sequence length.

## Step 6: Combine with compute-sanitizer

> ⚠️ `compute-sanitizer` is **NVIDIA-only**. On DCU/HCU (HIP/DTK), it is unavailable — use the local equivalent instead: `rocprof` / `rocgdb` for checkpointing, or the HIP/DTK memory-checking path via `dcu-vmfault-coredump` (VMFault faulting-kernel localization). Do not present compute-sanitizer output as evidence on a device that can't run it.

```bash
export SGLANG_KERNEL_API_LOGLEVEL=3
export SGLANG_KERNEL_API_LOGDEST=debug.log

compute-sanitizer --tool memcheck python3 /tmp/sglang_llm_crash.py
```

## Step 7: Kernel-Level Debugging with printf()

When you own the CUDA/HIP kernel, `printf()` is useful for narrowing down bad indices or launch geometry:

```cpp
__global__ void MyKernel(const float* input, float* output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("n=%d input0=%f\n", n, input[0]);
  }

  if (idx < n) {
    output[idx] = input[idx] * 2.0f;
  }
}
```

After launch, force the output to flush:

```python
my_kernel(...)
torch.cuda.synchronize()
```

### Warp-Specialized Kernels: Choosing the Right Print Thread

For warp-specialized kernels, `threadIdx.x == 0` only prints from the first warp. Use one representative lane per warp instead:

```cpp
// First lane of each warp
if ((threadIdx.x % warpSize) == 0) {
  printf("warp=%d val=%f\n", threadIdx.x / warpSize, local_val);
}
```

| Kernel Type | Print Condition | Notes |
|----------|----------|-------------|
| Simple kernel | `threadIdx.x == 0` | One thread per block is usually enough |
| Warp-specialized kernel | one representative lane per warp | e.g. `threadIdx.x % warpSize == 0` |
| Group-specialized kernel | one representative lane per group | choose based on the kernel's scheduling layout |

## Environment Variables Reference

| Variable | Values | Description |
|----------|--------|-------------|
| `SGLANG_KERNEL_API_LOGLEVEL` | `0` | No logging (default) |
|  | `1` | Function names only |
|  | `3` | Inputs and outputs with metadata |
|  | `5` | Level 3 plus tensor statistics |
|  | `10` | Level 5 plus crash-safe tensor dumps |
| `SGLANG_KERNEL_API_LOGDEST` | `stdout` | Log to stdout |
|  | `stderr` | Log to stderr |
|  | `<path>` | Log to file |
|  | `log_%i.txt` | `%i` expands to process ID |
| `SGLANG_KERNEL_API_DUMP_DIR` | `<path>` | Directory for level-10 dumps |
| `SGLANG_KERNEL_API_DUMP_INCLUDE` | wildcard list | Only dump matching API names |
| `SGLANG_KERNEL_API_DUMP_EXCLUDE` | wildcard list | Skip matching API names |

## Best Practices

**General verification:**

1. Prove that the candidate kernel was loaded and invoked; a correct final response does not prove path activation.
2. Validate shape, dtype, stride, device, pointer alignment, workspace lifetime, and stream ownership at the boundary.
3. Check exact required numerical semantics, NaN/Inf, and output IDs or hashes. Cosine similarity alone can hide severe element errors.
4. Re-run both single-rank and the target distributed configuration. A per-rank failure may surface as a collective hang elsewhere.
5. After fixing the crash, run the production concurrency and a soak test to catch allocator, KV-cache, or workspace reuse bugs.

**Kernel API log level:**

6. **Start with Level 3** - Usually enough to catch wrong shapes, dtypes, and devices.
7. **Use Level 5 for Numerical Issues** - When you suspect NaN or Inf values.
8. **Use Level 10 for Crash Reproduction** - Most useful when the process crashes before you can inspect live tensors.
9. **Log to File for Crashes** - `export SGLANG_KERNEL_API_LOGDEST=crash.log` - safer than stdout when the process aborts.
10. **Disable Logging in Production** - `unset SGLANG_KERNEL_API_LOGLEVEL`

## Troubleshooting

### No Logs Appear
Check: `echo $SGLANG_KERNEL_API_LOGLEVEL`, `echo $SGLANG_KERNEL_API_LOGDEST`, and whether the failing path goes through a covered API boundary.

### Too Much Output
Reduce the level: `export SGLANG_KERNEL_API_LOGLEVEL=3`

### Statistics/Tensor Dumps Skipped During CUDA Graph Capture
Expected behavior. Level-5 statistics and level-10 dumps require copying tensors to CPU, which is not allowed during CUDA graph capture.

## 验收

1. 复现是**同步**的（每次都在同一位置崩）。
2. 调用前日志能定位到具体的 kernel 启动参数。
3. 张量校验排除了 NaN/Inf 与越界形状。
4. 结论指出是自定义算子、框架内核、还是调用方传参。

## 不适用场景

分布式卡住而非崩溃（走 `debug-distributed-hang`）；
VMFault 且已有 coredump（走 `dcu-vmfault-coredump`）；
问题不可复现（先解决可复现性，否则任何结论都不可信）。
