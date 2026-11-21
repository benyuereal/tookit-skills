# Framework Tuning Playbooks（自 SKILL.md Workflow §4-6 下沉）

> 2026-10-16 自 `perf/llm-serving-auto-benchmark/SKILL.md` 下沉（SKILL.md 瘦身到 500 行内）。
> 三框架逐框架搜索轴/护栏/参数族原文照录；工作流骨架（preflight/normalize/search tier）仍在主 SKILL.md。

### 4. Tune SGLang

Prefer the SGLang auto-benchmark runner when the target checkout supports it:

```bash
python -m sglang.auto_benchmark run --config /path/to/sglang.yaml
```

Otherwise launch the server manually and benchmark with:

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 256 \
  --num-prompts 80 \
  --request-rate 8 \
  --output-file /path/to/sglang/results.json \
  --output-details
```

Version-sensitive SGLang knob families to verify:

- `tp_size`, `pp_size`, `dp_size`, `ep_size`
- `attention_backend`, `prefill_attention_backend`, `decode_attention_backend`
- `sampling_backend`
- `max_running_requests`, `max_queued_requests`
- `chunked_prefill_size`, `prefill_max_requests`, `max_prefill_tokens`
- `max_total_tokens`, `page_size`
- CUDA graph and piecewise CUDA graph settings
- speculative or EAGLE settings only after the non-speculative baseline is tuned

Keep `mem_fraction_static` and `schedule_policy` pinned in the default pass,
matching the shared cookbook config style.

For quick smoke tests, it is reasonable to disable CUDA graph and piecewise CUDA
graph startup work if the goal is only to prove the framework flow. Record those
flags in the artifact. Do not carry that smoke setting into a performance winner
unless the user asked to tune eager-mode serving.

### 5. Tune vLLM

Use vLLM's sweep runner when available:

```bash
vllm bench sweep serve \
  --serve-cmd 'vllm serve <model> --port 8000' \
  --bench-cmd 'vllm bench serve --backend vllm --model <model> --port 8000 --dataset-name random --num-prompts 80' \
  --serve-params /path/to/vllm_serve_params.json \
  --bench-params /path/to/vllm_bench_params.json \
  --output-dir /path/to/vllm_results
```

If sweep support is unavailable, run `vllm serve` for each candidate and measure
with `vllm bench serve`.

Version-sensitive vLLM knob families to verify:

- tensor, pipeline, data, decode-context, and expert parallelism
- `gpu_memory_utilization`
- `max_num_seqs`
- `max_num_batched_tokens`
- `max_model_len`
- `enable_chunked_prefill`, partial prefill limits, and DBO thresholds
- KV cache dtype and block size
- dtype and quantization settings
- CUDA graph capture sizes or eager-mode toggles when relevant
- prefix cache and speculative decoding settings only when the workload needs
  those features

vLLM should get a normal sweep, not one baseline command. See
[references/framework-reference.md](framework-reference.md) for
native command templates and cross-framework knob families. Confirm each flag on
the target image's `--help` before a run.

Keep `gpu_memory_utilization` in the baseline for the default pass. Search it
only when the question is explicitly about fitting the model or trading capacity
against throughput.

Keep DBO and all2all backend settings out of the default pass unless the target
vLLM environment is already set up for them. They are real tuning knobs, but a
candidate can fail at startup if the required all2all backend is not available.
Also preflight concurrent partial prefill before raising
`max_num_partial_prefills` above 1; some model/runtime combinations reject it at
startup.

### 6. Tune TensorRT-LLM

Use `trtllm-serve serve` as the server entrypoint when the target environment
supports it:

```bash
trtllm-serve serve <model> \
  --backend pytorch \
  --tp_size <tp> \
  --pp_size <pp> \
  --kv_cache_free_gpu_memory_fraction 0.75 \
  --host 0.0.0.0 \
  --port 8000
```

Then benchmark the OpenAI-compatible endpoint with the TensorRT-LLM serving
benchmark client or with the same OpenAI-compatible client used for the other
frameworks.

In the historical TensorRT-LLM 1.0.0 validation image,
`benchmark_serving --dataset-name random` sampled from ShareGPT unless either
`--download-path` or `--random-ids` was passed. For a fast synthetic smoke test,
pass `--random-ids`, then confirm the behavior on the target TensorRT-LLM image.

TensorRT-LLM flag names are especially version-sensitive. In the validated
TensorRT-LLM 1.0.0 image, the KV-cache memory flag accepted by
`trtllm-serve serve` was `--kv_cache_free_gpu_memory_fraction`, not
`--free_gpu_memory_fraction`. Current mainline was rechecked at `7021547` on
2026-05-15; the newer commits did not touch serving code, so the `b9e1945`
serving flag evidence still applies: both aliases are accepted and
`trtllm-serve` still defaults to the PyTorch backend. Always verify flags with
`trtllm-serve serve --help` before running a search on any GPU target.

TensorRT-LLM backend policy for this skill:

- launch the server with `--backend pytorch`
- keep `backend: pytorch` in `base_server_flags`
- do not add `backend` to `search_space`
- reject `trt`, engine-backed serving, or any other non-PyTorch TensorRT-LLM
  server backend as unsupported for this skill

Version-sensitive TensorRT-LLM knob families to verify:

- `tp_size`, `pp_size`, and `ep_size`
- max batch size, max sequence length, max number of tokens, and KV-cache budget
- inflight batching and scheduler options
- extra LLM API options YAML used by `trtllm-serve` with the PyTorch backend

The `trtllm-serve serve` CLI exposes fewer direct runtime knobs than SGLang or
vLLM. Use direct flags when they exist, then use `--extra_llm_api_options` for
PyTorch-backend settings that are not top-level CLI flags. Keep unsupported
backend or engine requests in the failure table instead of translating them.

Keep `kv_cache_free_gpu_memory_fraction` in the baseline for the default pass.
Search `max_batch_size`, `max_num_tokens`, `max_seq_len`, and validated
PyTorch-backend config options first. The server backend remains fixed to
`pytorch`.
