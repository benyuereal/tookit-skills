---
name: generate-profile
description: Capture reproducible SGLang or vLLM serving profiles on CUDA, HIP, DCU, or HCU targets. Use when collecting graph-off source-mapping traces, graph-on production traces, prefill/decode profiles, rank-specific traces, or evidence for an LLM inference bottleneck. Enforce correctness, exact process ownership, remote-environment checks, and artifact provenance.
version: 1.0.0
author: DCU Engineer
license: MIT
kind: knowledge
---

# Generate an LLM Serving Profile

Capture evidence from the actual target path. Never kill processes by name or profile an
unverified fallback configuration.

## Establish the capture contract

Record before launch:

- Framework commit, model/checkpoint, precision, TP/PP/DP/EP, workload and target rank.
- GPU architecture, DTK/CUDA/ROCm version, container identity and relevant environment.
- Whether the trace is graph-off mapping evidence or graph-on production evidence.
- Output directory outside the repository for large traces.

On DCU/HCU, probe the environment rather than assuming NVIDIA behavior. Preserve required
platform settings such as `AMDGCN_USE_BUFFER_OPS=0`. Use eager mode for a mapping trace or when
the platform contract requires it; enable graphs only when graph behavior is the subject under
test and correctness has already passed.

## Verify the target

1. Check GPU processes, utilization, memory, health and VMFault/RAS state.
2. Confirm no co-tenant workload invalidates the measurement window.
3. Verify the selected attention, MoE, quantization and graph paths from logs or maps.
4. Send a deterministic correctness request and save output IDs, token counts and response data.
5. Stop if the service fell back, produced NaN/Inf, returned the wrong token count, or cannot
   prove which implementation ran.

## Launch safely

Prefer profiling an existing user-identified service. If launching a new service, save its exact
PID or container name:

```bash
profile_dir=/tmp/llm-profile-$(date +%Y%m%d-%H%M%S)
mkdir -p "$profile_dir"

AMDGCN_USE_BUFFER_OPS=0 CUDA_VISIBLE_DEVICES=<ids> \
  sglang serve --model-path <model> --port <port> <validated-args> \
  >"$profile_dir/server.log" 2>&1 &
server_pid=$!
printf '%s\n' "$server_pid" >"$profile_dir/server.pid"
```

Poll readiness with a bounded timeout. On failure, preserve the log and inspect the exact PID.
Do not use broad `pkill`, `killall`, container restart, or GPU reset.

## Capture two complementary traces

Use framework-native profiling controls. For SGLang, use the installed version's supported
`sglang.test.send_one --profile` or profile HTTP endpoints; for vLLM, launch with its profiler
configuration. Verify the CLI/API against the installed commit before invoking it.

### SGLang-specific controls (verified on 0.5.12, 2026-09-12)

- Trace output directory: `SGLANG_TORCH_PROFILER_DIR=<dir>` (default `/tmp`). Set it explicitly
  to a path outside the repository; when driving `/start_profile` manually without an
  `output_dir` field, traces land in this directory.
- Endpoints: `POST /start_profile` and `POST /stop_profile` (both also accept GET). The
  `/start_profile` request body accepts `output_dir`, `num_steps`, `activities`,
  `profile_by_stage`, `merge_profiles`, `profile_prefix`, `start_step`.
- Multi-TP merge: send `merge_profiles: true` in the `/start_profile` body to merge per-rank
  traces into a single JSON for cross-rank comparison. Supported on the default v1 profiler
  path (`SchedulerProfilerMixin._merge_profile_traces`). NOT supported on the v2 path
  (`SGLANG_PROFILE_V2=1`, `ProfileManager.configure` asserts `not merge_profiles` and
  `profile_by_stage`) — v2 currently only supports stage-based profiling.
- Graph-off mapping trace: launch the server with `--disable-cuda-graph`. Do NOT use
  `SGLANG_CUDA_GRAPH_MODE=0` / `SGLANG_CACHE_GRAPH=0` — these environment variables do not
  exist in SGLang 0.5.12 (verified by full source grep); they are a circulating tutorial
  error and have no effect.

Capture when the question requires source mapping:

1. **Mapping trace**: graph off, short active window, source stacks enabled.
2. **Formal trace**: production graph/backend settings, same workload and warm state.

For prefill/decode questions, separate the workloads. Capture one rank first; collect all ranks
only for imbalance, collective or hang analysis. Avoid merging rank traces before checking clock
alignment.

## Validate and clean up

After capture:

- Repeat the deterministic correctness request.
- Record wall-clock TTFT/TPOT alongside the trace; profiler time can be inflated.
- Save server args, environment allowlist, workload, output hash and trace filenames.
- Stop only the process created by this run:

```bash
kill "$(cat "$profile_dir/server.pid")"
```

If graceful termination times out, inspect ownership and descendants before escalating. Never
terminate unrelated serving processes.

Return the trace paths, capture mode, rank, workload, correctness result, wall-clock metrics and
any fallback or contamination warnings.
