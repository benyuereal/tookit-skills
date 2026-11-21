# 真实 H100 验证（跨平台对照证据）

> NVIDIA H100 平台实测记录，用于证明工具跨平台可用；DCU 实测结论以正文各节为准。
> 2026-10-16 自 `perf/llm-torch-profiler-analysis/SKILL.md` 下沉（SKILL.md 瘦身到 500 行内），内容原文照录。

## 真实 H100 验证

> 注：本节为上游 H100 验证记录（非 DCU 基线）；文中 `/data/bbuf/...` 为特定主机路径，非仓库内。

当前参考运行是在 `2026-04-23` 于 `h100_sglang` 上捕获的 `4x H100` 矩阵：

- `/data/bbuf/validate/unified_llm_profiler_skill/runs/20260423_h100_large_model_matrix_v3`

渲染后的 markdown bundle：

- `/data/bbuf/validate/unified_llm_profiler_skill/runs/20260423_h100_large_model_matrix_v3/h100_large_model_matrix_v3_bundle.md`

已验证的模型目录：

- `mixtral_8x7b_instruct`
- `qwen2_5_32b_instruct`
- `qwen3_32b`

每个模型目录包含：

- `analysis_sglang.txt`
- `analysis_vllm.txt`
- `analysis_trtllm.txt`
- 框架相关的 trace roots 和 probe artifacts

验证矩阵：

| Model | SGLang | vLLM | TensorRT-LLM | Result |
| --- | --- | --- | --- | --- |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | `4x H100` | `4x H100` | `4x H100` | 三张表在三个框架上都能正确渲染；benchmark probe 返回直接且非空文本 |
| `Qwen/Qwen2.5-32B-Instruct` | `4x H100` | `4x H100` | `4x H100` | 三张表在三个框架上都能正确渲染；benchmark probe 返回直接且非空文本 |
| `Qwen/Qwen3-32B` | `4x H100` | `4x H100` | `4x H100` | 三张表在三个框架上都能正确渲染；vLLM 和 TensorRT-LLM chat probe 经常输出 `<think>` 前缀 |

把这个运行作为主要 H100 参考。
较旧的 `2026-04-22` 单卡 Qwen3 矩阵仍然可用于 bring-up，但已经不是默认参考。

阶段拆分 workload 验证于 `2026-05-01` 在 `h100_sglang` 上捕获：

- `/data/bbuf/validate/unified_llm_profiler_skill/runs/20260501_stage_split_validation`
- `/data/bbuf/validate/unified_llm_profiler_skill/runs/20260501_stage_split_validation_large`

已验证模型：

| Model | GPU | Workloads | Result |
| --- | --- | --- | --- |
| `Qwen/Qwen2.5-0.5B-Instruct` | `1x H100` | prefill `4090->1`, decode `1->2048` | 生成了独立的 `prefill/*.trace.json.gz` 和 `decode/*.trace.json.gz`；kernel、overlap、fuse 表按 `extend/prefill` 和 `decode` 分段渲染 |
| `Qwen/Qwen2.5-1.5B-Instruct` | `1x H100` | prefill `4090->1`, decode `1->2048` | 生成了独立的 `prefill/*.trace.json.gz` 和 `decode/*.trace.json.gz`；kernel、overlap、fuse 表按 `extend/prefill` 和 `decode` 分段渲染 |
| `Qwen/Qwen2.5-7B-Instruct` | `1x H100` | prefill `4090->1`, decode `1->2048` | 生成了独立 trace；prefill kernel table 捕获 28 层 GEMM/FA3/RMSNorm 工作，decode 捕获 5-step graph launches，fuse rows 按 stage 拆分 |
| `Qwen/Qwen2.5-14B-Instruct` | `1x H100` | prefill `4090->1`, decode `1->2048` | 生成了独立 trace；prefill kernel table 捕获 48 层 GEMM/FA3/RMSNorm 工作，decode 捕获 5-step graph launches，fuse rows 按 stage 拆分 |
| `Qwen/Qwen3-8B` | `2x H100`, TP=2 | prefill `4090->1`, decode `1->2048`, warmup 10/capture 5 | 生成了独立 prefill/decode trace 和全部三张表；唯一 probe prompt 避免了 prefill 表中的 prefix-cache 污染 |
| `mistralai/Mistral-7B-Instruct-v0.3` | `2x H100`, TP=2 | prefill `4090->1`, decode `1->2048`, warmup 10/capture 5 | 生成了独立 prefill/decode trace 和全部三张表；server logs 显示 active prefill 窗口中没有重复 prompt prefix-cache 快捷路径 |

这次验证也覆盖了旧版 SGLang profiler 状态机的兼容修复：
workload 分离的 live capture 使用输出目录标记 stage，并避免在每个 workload 内嵌套 SGLang 内部的 `profile_by_stage` 状态机。
该 helper 会增加一个内部 scheduler guard step，因为 SGLang 会先递增 `forward_ct`，再检查 profiler 是否应该停止；如果没有这个 guard，`num_steps=1` 的 prefill capture 可能会在真正的 prefill forward 之前就停止。

2026-05-01 两卡验证中额外模型的 artifacts 为：

- `/data/bbuf/validate/core_skill_validation_20260501/qwen3_8b/profiler`
- `/data/bbuf/validate/core_skill_validation_20260501/mistral_7b_instruct_v03/profiler`

将已验证运行渲染成一个 markdown 文档：

```bash
python3 scripts/render_triage_markdown_bundle.py \
  --analysis-root /data/bbuf/validate/unified_llm_profiler_skill/runs/20260423_h100_large_model_matrix_v3 \
  --output /data/bbuf/validate/unified_llm_profiler_skill/runs/20260423_h100_large_model_matrix_v3/h100_large_model_matrix_v3_bundle.md
```

bundle 按模型分组，并为每个框架保留三张表。

H100 备注：

- 当 trace 包含干净的 stage split 时，三个框架现在都能按独立的 `extend/prefill` 和 `decode` section 渲染 kernel、overlap 和 fuse 表
- SGLang live capture 已验证，并且直接调用 server profiler API，而不是 shell 到 `sglang.profiler`
- SGLang trace flush 可能比几秒钟长很多，因此 runner 等待 artifacts 的时间比早期实现更长
- SGLang kernel-site reconstruction 在 mapping 路径中保持 sampling disabled，这样优化后的 parser 不会扰动 SGLang 表输出；`Mixtral-8x7B-Instruct-v0.1`、`Qwen3-32B` 和 `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` 的 equality recheck 已匹配
- vLLM live capture 要求 `--output-dir` 与 server 的 `torch_profiler_dir` 一致；已验证的 H100 流程使用 `--profiler-config {"profiler":"torch","torch_profiler_dir":"..."}`，随后驱动 `/start_profile` 和 `/stop_profile`
- TensorRT-LLM 验证保持在 `--backend pytorch`；H100 流程通过 `TLLM_TORCH_PROFILE_TRACE` 写 trace，然后分析保存的 trace
- TensorRT-LLM 当前 mainline 已于 `2026-05-15` 在 `7021547` 重新检查；更新提交没有触碰 profiling/server 代码，因此 `b9e1945` profiler 证据仍然适用：PyTorch profiling 使用 `record_shapes=True` 和 `with_modules=True`，但不使用 `with_stack=True`；除非目标镜像证明不需要，否则保留用于表格质量 Python location 的 override 路径
- 在这台 host 上，把所有 trace roots 放在 `/data/...` 下，而不是 `/home/...`
