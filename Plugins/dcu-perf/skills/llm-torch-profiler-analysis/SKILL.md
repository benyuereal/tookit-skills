---
name: llm-torch-profiler-analysis
description: "面向 `sglang`、`vllm` 和 `TensorRT-LLM` 的统一 LLM torch-profiler 分析 skill。用于检查已有的 `trace.json(.gz)` 或 profile 目录，也可以驱动运行中的服务进行 live profiling，并返回一个包含 kernel、overlap-opportunity、fuse-pattern 和空泡分析的分析文档。"
version: 1.0.0
author: DCU Engineer
license: MIT
kind: tool
---

# 统一 LLM Torch Profiler 分析

## 概览

使用这个 skill 分析以下框架的 `torch.profiler` 结果：

- `sglang`
- `vllm`
- `TensorRT-LLM`

公开工作流只有一个：

- `triage`

推荐的统一入口：

- [scripts/analyze_llm_torch_profile.py](scripts/analyze_llm_torch_profile.py)

推荐的报告入口：

- [scripts/run_triage_report.py](scripts/run_triage_report.py)

除非用户明确只要 stdout，已有 trace/profile 的分析默认使用 `run_triage_report.py`。
它会在用户指定的工作目录下同时生成：

- `terminal_commands.log`：保存本次 skill 在终端实际运行的脚本指令、时间和 cwd
- `analysis_stdout.txt`：保存 `analyze_llm_torch_profile.py` 的原始输出
- `analysis_stderr.txt`：当分析脚本有 stderr 时保存错误/警告输出
- `torch_profiler_analysis.md`：最终 Markdown 分析报告

最终报告必须写入指定工作目录，不能只在聊天中返回表格。
如果用户没有指定工作目录，先使用 trace/profile 目录下的 `profiler_analysis_report/`。
如果 trace/profile 目录不可写，使用 `/tmp/llm_torch_profiler_analysis/<模型或trace目录名>/`。

最终报告必须包含模型结构图章节。使用 `model-architecture-diagram` skill 的
`../model-architecture-diagram/scripts/model_architecture_diagram.py "<model name>"`（跨 skill 引用：该脚本在 `model-architecture-diagram/scripts/` 下，非本 skill 内）生成公开原始模型结构图 Markdown，
并嵌入 `torch_profiler_analysis.md`。如果没有匹配到公开原始结构图，也要在报告里记录
`no_match` 结果，不能静默省略。

向后兼容 shim，保留它是为了让旧的 `docker exec ... analyze_sglang_torch_profile.py ...` 调用继续可用；它只是转发到统一入口：

- [scripts/analyze_sglang_torch_profile.py](scripts/analyze_sglang_torch_profile.py)

Markdown bundle 渲染辅助工具：

- [scripts/render_triage_markdown_bundle.py](scripts/render_triage_markdown_bundle.py)

`triage` 总是打印同样三张表，并在同一份分析文档里追加一个空泡分析章节：

- kernel table
- overlap-opportunity table
- fuse-pattern table

空泡分析章节会分别分析 `prefill` 和 `decode`，同时输出 `Global idle bubble` 和 `Per-stream gap`，并在章节开头注明 trace 目录。gap 表会按相同前后 kernel 位置合并，且至少展示 20 行。

默认情况下，三张表都只渲染累计 GPU 时间占比大于等于 `1.0%` 的行。
低于该阈值的行默认隐藏，除非用户要求使用更低 cutoff。

保持 fuse-pattern table 基于源码证据并且确定可复现。
不要把它变成模糊匹配器。

如果精确的源码证据匹配较弱，但某个 kernel cluster 仍然接近已知模式族，
在表格后追加一条简短 note，且只能使用下面三个值之一：

- `high`
- `medium`
- `low`

## 能力矩阵

| 能力 | SGLang | vLLM | TensorRT-LLM |
| --- | --- | --- | --- |
| 已有 trace triage | yes | yes | yes |
| 单 trace live capture | yes | yes, if torch profiler is enabled on server | requires profiler control endpoints |
| 双 trace mapping+formal triage | yes | yes | yes |
| 阶段拆分的 live workload | yes | yes | yes, with a writable shared trace dir or per-stage host runner |
| `--profile-by-stage` capture | yes | no | no |
| `--profile-prefix` control | yes | usually ignored on HTTP profiler route | usually ignored on HTTP profiler route |

对于 TensorRT-LLM，live capture 只有在服务暴露 `/start_profile` 和
`/stop_profile`，并且部署已经提供共享 trace 路径和所需环境变量时才可用。

## 真实 H100 验证（已下沉）

H100 平台实测过程与结论（证明工具跨平台可用的对照证据）移至 [references/h100-validation.md](references/h100-validation.md)。

## 何时使用

- 检查来自 `sglang`、`vllm` 或 `TensorRT-LLM` 的 `torch.profiler` trace 或 profile 目录
- profile 一个 live serving endpoint 并分析结果
- 总结 prefill 或 decode 阶段由哪些 kernel family 主导
- 将 kernel 映射回 Python 代码路径
- 判断某条代码路径是否仍有 overlap 机会
- 检查某个已知 fusion 或 overlap 路径是否本应生效

## Diffusion Backend Gate

对于 diffusion benchmark 或 profiling 工作，只分析由 SGLang native diffusion backend 产生的 trace。

如果生成 trace 的运行日志包含以下任意内容：

- `Falling back to diffusers backend`
- `Using diffusers backend`
- `Loaded diffusers pipeline`

停止 workflow，不要分析该 trace。
把它作为 backend-selection 问题处理，而不是 native-kernel profiler 证据。

## 主流程

## Stage-Separated Live Capture Contract

live capture 默认不能使用一个混合 prompt。
默认情况下，`analyze_llm_torch_profile.py --url ...` 会捕获两个带标签的 workload，
然后用独立的 stage section 渲染同样三张表：

- prefill: synthetic input length `4090`, output length `1`
- decode: synthetic input length `1`, output length `2048`

每条 live profiler 路径默认先 warm up `10` 步，再 arm profiler，随后 capture `5` 个 active step。
在比较 kernel 表之前，要在 SGLang、vLLM 和 TensorRT-LLM 之间保持这个 warmup/active split 一致。

当已知 benchmark workload 时，使用这些选项覆盖默认 contract：

```bash
--profile-workload both \
--warmup-steps 10 --num-steps 5 \
--prefill-input-len 4090 --prefill-output-len 1 \
--decode-input-len 1 --decode-output-len 2048
```

允许的 `--profile-workload` 取值：

- `both`: 默认；分别捕获 prefill 和 decode
- `prefill`: 只捕获长输入 / 单 token 输出 workload
- `decode`: 只捕获单 token 输入 / 长输出 workload
- `legacy`: 保留旧的 `--probe-prompt` / `--probe-max-new-tokens` 行为

对于 `sglang-sota-humanize-loop`，如果慢的 SGLang benchmark 场景已有已知输入/输出分布，不要使用默认值。
应使用该慢场景设置 profiler 长度：prefill 使用慢输入长度并输出 `1`，decode 使用输入 `1` 并输出慢输出长度。
对于混合数据集，profile benchmark 报告中已有的最慢代表性 bucket，例如 p50 或 p95 输入/输出对，并在 artifact notes 中记录该 bucket。

### 1. 从已有 profile dir 或 trace 做单 trace triage

默认使用报告入口，并显式传入工作目录和模型名：

```bash
python3 scripts/run_triage_report.py \
  --work-dir /path/to/work_dir \
  --input /path/to/profile_dir_or_trace.json.gz \
  --framework auto \
  --model MiniMax-M2.5
```

该命令会把实际执行的 `analyze_llm_torch_profile.py` 命令写入
`/path/to/work_dir/terminal_commands.log`，并生成
`/path/to/work_dir/torch_profiler_analysis.md`。报告中会包含
`model-architecture-diagram` 返回的模型结构图 Markdown。

只有在调试脚本本身、或用户明确要求只看原始 stdout 时，才直接运行底层入口：

```bash
python3 scripts/analyze_llm_torch_profile.py \
  --input /path/to/profile_dir_or_trace.json.gz
```

当一个 trace 已经足够时使用。
在 single-trace 模式下，overlap table 会保持保守，并提示何时需要 mapping/formal pair。

### 2. 从 SGLang 做单 trace live capture

```bash
python3 scripts/analyze_llm_torch_profile.py \
  --framework sglang \
  --url http://127.0.0.1:30000 \
  --output-dir /data/bbuf/validate/unified_llm_profiler_skill/runs/example/sglang_profile_live \
  --num-steps 5 \
  --warmup-steps 10 \
  --profile-by-stage \
  --profile-workload both
```

脚本会直接向 SGLang server 发送 `POST /start_profile`（请求体带 `output_dir`，server 按该目录落 trace；
server 端默认目录由 `SGLANG_TORCH_PROFILER_DIR` 控制，默认 `/tmp`——手动 curl `/start_profile` 不带
`output_dir` 时 trace 会落到 `/tmp`，注意别漏收）。
保持 `--output-dir` 位于 `/data/...` 下，这样后续分析和文档可以看到 trace。
脚本会写入 `server_args.json`，使用同样 workload shape 进行 warmup，在 profiling armed 后发送 active probe requests，默认捕获独立的 `prefill/` 和 `decode/` profile roots，并且比早期实现等待更长 trace flush 时间。
对于默认的 workload-separated capture，目录名标记 stage，而且每个 workload 内不使用 SGLang 内部 `profile_by_stage` 模式。
这样可以避免把一个单 token prefill probe 和独立 decode profile 混在一起。
该 helper 仍然会增加一个内部 guard step，因为旧版 SGLang profiler 会在执行下一次 forward 之前检查目标计数器。

### 3. 从 vLLM 做单 trace live capture

启动启用 torch profiler 的 vLLM，例如：

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/data/bbuf/validate/unified_llm_profiler_skill/runs/example/vllm_profile"}'
```

然后运行：

```bash
python3 scripts/analyze_llm_torch_profile.py \
  --framework vllm \
  --url http://127.0.0.1:8000 \
  --output-dir /data/bbuf/validate/unified_llm_profiler_skill/runs/example/vllm_profile \
  --num-steps 5 \
  --warmup-steps 10 \
  --no-profile-by-stage \
  --profile-workload both
```

对于 vLLM，`--output-dir` 必须指向 server 使用的同一个 `torch_profiler_dir`。
当前 vLLM profiler config 已经默认 `torch_profiler_with_stack=true`，
因此 runner 只需要设置 `torch_profiler_dir`。
在 `h100_sglang` 上，外部 vLLM containers 应同时挂载：

- `/data/.cache/huggingface:$HOME/.cache/huggingface (本机 HF 缓存，按需替换)`
- `/data/bbuf/validate/unified_llm_profiler_skill:/data/bbuf/validate/unified_llm_profiler_skill`

### 4. 从 TensorRT-LLM 做单 trace live capture

只有当 server 暴露 `POST /start_profile` 和 `POST /stop_profile`，
并且 trace 路径与当前机器共享时，才使用此流程。

典型环境变量预期：

- `TLLM_PROFILE_START_STOP=<start>-<stop>`，例如 `10-20`
- `TLLM_TORCH_PROFILE_TRACE=/shared/path/trace.json` 或 `.json.gz`

然后运行：

```bash
python3 scripts/analyze_llm_torch_profile.py \
  --framework trtllm \
  --url http://127.0.0.1:8000 \
  --output-dir /shared/path \
  --num-steps 5 \
  --no-profile-by-stage \
  --profile-workload both
```

如果部署没有暴露 profiler control endpoints，回退为分析已有 trace，而不是尝试 live capture。
如果 TensorRT-LLM trace output 配置为一个固定文件路径，使用
`scripts/run_trtllm_pytorch_profile_host.sh --stage prefill` 和 `--stage decode`，
而不是直接使用 `--profile-workload both`，这样每个 stage 都会获得自己的 trace 文件。

在当前 TensorRT-LLM mainline 路径中，`py_executor.py` 创建 torch profiler 时使用
`record_shapes=True` 和 `with_modules=True`，但不使用 `with_stack=True`。
为了获得表格质量的验证，使用 override generator：

```bash
python3 scripts/make_trtllm_py_executor_override.py \
  --source /path/to/original/py_executor.py \
  --output /data/bbuf/validate/unified_llm_profiler_skill/overrides/trtllm/py_executor_with_stack.py
```

matrix runner 在 H100 上启动 TensorRT-LLM capture 之前会自动执行这一步。

这是在 `h100_sglang` 上验证过的 TensorRT-LLM 流程：

1. 使用 `TLLM_PROFILE_START_STOP=<start>-<stop>` 和 `TLLM_TORCH_PROFILE_TRACE=/data/.../trace.json` 启动 `trtllm-serve`
2. 运行几个 benchmark requests
3. 使用 `--input /data/.../trace.json` 分析生成的 trace

### 5. 从已有 profile dirs 或 traces 做双 trace triage

```bash
python3 scripts/analyze_llm_torch_profile.py \
  --mapping-input /path/to/graph_off_profile_dir \
  --formal-input /path/to/graph_on_profile_dir
```

当你需要更强的 overlap attribution 和 kernel-to-source mapping 时使用。

### 6. 从运行中的 servers 做双 trace triage

```bash
python3 scripts/analyze_llm_torch_profile.py \
  --framework sglang \
  --mapping-url http://127.0.0.1:31025 \
  --formal-url http://127.0.0.1:31026 \
  --num-steps 5 \
  --profile-by-stage
```

对于 `vllm` 或 `TensorRT-LLM`，使用同样形状，但传入：

- `--framework vllm` 或 `--framework trtllm`
- `--mapping-output-dir ...`
- `--formal-output-dir ...`
- `--no-profile-by-stage`

## `profile_by_stage`

`--profile-by-stage` 只在 SGLang live-capture 路径上有意义。

- 使用 `--profile-workload both` / `prefill` / `decode` 时，workload 目录就是 stage 标签；live-capture helper 会按 workload 禁用 SGLang 内部 stage profiler，先 warmup，再为选中的 workload 捕获请求的 active step 数。
- 对于 legacy 或手动捕获的 SGLang serving，内部 `profile_by_stage` 仍然有用，因为 prefill 和 decode 通常有非常不同的瓶颈。
- 在当前 SGLang 内部 profile-v2 路径上，stage-based profiling 实际上就是常规路径。
- PD-disaggregated serving 增加一条额外规则：prefill workers 和 decode workers 必须分别 profile。这比普通 `profile_by_stage` 更严格。
- 对于 `vllm` 和 `TensorRT-LLM`，使用 `--no-profile-by-stage` 禁用它。

## 如何选择 Triage 形态

### 单 trace triage

当你想要最低摩擦的报告时使用：

- 已经有一个 trace
- 主要想看 kernel share 和 fusion 线索
- 通过分别运行一次 triage 来并排比较两个 runs

默认优先使用这种方式。

### 双 trace triage

当你需要以下能力时使用：

- 更强的 overlap 答案
- graph-off 源码映射加 graph-on 最终行为
- 中间表中更可信的 overlap recommendations

1. mapping trace 使用 graph disabled，或者使用 lower-fusion / 更易读的配置
2. formal trace 使用真实 serving optimizations enabled

不要把 mapping pass 称为 "fast profile"。
它的用途是恢复 `kernel -> cpu_op -> python scope`。

## 工作流

### 单 trace 工作流

1. 如果用户只需要诊断，一个 trace 就足够。
2. 必须先确定工作目录，并把所有终端脚本指令写入该目录下的 `terminal_commands.log`。
3. 必须使用 `scripts/run_triage_report.py` 生成最终 `.md` 报告，除非用户明确要求只输出 stdout。
4. 最终报告必须包含 `model-architecture-diagram` 生成的模型结构图章节。
5. 只要 profiler 同时产生了 one-rank traces 和 merged traces，就优先使用 one-rank traces。
6. 对于 live server，只有在框架特定的前置条件已经满足时，才让脚本驱动 profiler。
7. 优先使用 `--profile-workload both`；只有在复现旧 trace contract 时才使用 `legacy`。
8. 优先使用 workload-separated SGLang capture；内部 `--profile-by-stage` 主要用于 `legacy` 或手动收集的 traces。
9. 在 `h100_sglang` 上，通过 `docker exec sglang_bbuf ...` 创建或清理目标 trace 目录，这样可以确保该路径在 `/data` 下可写。

### 双 trace 工作流

1. 先生成 mapping trace，使用 graph disabled 或 lower-fusion 配置。
2. 再生成 formal trace，使用真实 serving optimizations enabled。
3. 运行 `triage` 生成三张表报告。
4. 按以下顺序阅读结果：
   - kernel table
   - overlap-opportunity table
   - fuse-pattern table
5. 在把某个东西称为“新”优化想法之前，将 top rows 同时与 [references/fuse-overlap-catalog.md](references/fuse-overlap-catalog.md) 和 [references/overlap-catalog.md](references/overlap-catalog.md) 对比。先检查 mainline rows，再检查 `PR-backed / in-flight` sections。优先报告：
   - 一个本应已在这里适用的 existing fused 或 overlap path
   - 一个在当前 trace 中看起来 disabled、unsupported 或 regressed 的 existing path
   - 一个在其他地方已是 mainline 但本地缺失的 upstream pattern，或者仍然 open upstream 的 pattern
   - 只有当没有 catalog entry 符合时，才报告真正的新机会
6. 如果没有完全匹配的精确 pattern，但 trace 仍然接近某个已知 family，在表后添加一条平铺的 similarity note。
   只能使用 `high`、`medium` 或 `low`。
   基于完整 pattern shape，而不是只基于一个 kernel name。
   优先使用 producer-consumer chain、source locations、CPU op names、TP context 和 model-specific structure 等语义线索。
   不要重写脚本表本身来加入这些启发式判断。

## SGLang Pitfalls（2026-09-12 在 0.5.12 实机源码验证）

1. **关 graph 做 mapping trace 用 `--disable-cuda-graph` 启动参数**。部分教程流传的
   `SGLANG_CUDA_GRAPH_MODE=0` / `SGLANG_CACHE_GRAPH=0` 环境变量在 SGLang 0.5.12 源码中
   不存在（全源码 grep 0 命中），设了无效。
2. **FlashInfer JIT 假异常**：首个 prefill 请求触发 FlashInfer JIT 编译，layer 0 耗时异常长。
   这是编译开销不是性能问题；默认 warmup 10 步已覆盖，分析 kernel table 时不要把 layer 0
   的异常耗时当瓶颈。
3. **sampler 不在 CUDA graph 内**：decode trace 里 sampler 算子以独立 launch 出现在 graph
   replay 之外，属正常现象，不是 graph 失效。
4. **`merge_profiles: true`** 可把多 TP 的 per-rank trace 融合成单个 JSON 便于跨 rank 对照。
   默认 v1 profiler 路径（`SchedulerProfilerMixin`）支持；`SGLANG_PROFILE_V2=1` 的 v2 路径
   （`ProfileManager`）会 assert 禁止（v2 目前只支持 `profile_by_stage` 模式）。

## References

只在需要时加载这些文件：

- [references/source-map.md](references/source-map.md)
  - upstream SGLang profiler entrypoints 和 trace-writing paths；对 SGLang-specific source follow-up 仍然最有用
- [references/heuristics.md](references/heuristics.md)
  - overlap labels、dependency-risk interpretation 和 limits
- [references/fuse-overlap-catalog.md](references/fuse-overlap-catalog.md)
  - 混合的 source-backed catalog，包含 existing fuse 和 overlap patterns，也包括 mainline rows 以及 PR-backed / in-flight rows
- [references/vllm-torch-compile-fusions.md](references/vllm-torch-compile-fusions.md)
  - 当前 vLLM torch.compile fusion passes 及其目标 source patterns
- [references/overlap-catalog.md](references/overlap-catalog.md)
  - 跨 LLM、VLM、diffusion、disaggregation、HiSparse 和 speculative scheduling 的 overlap-only lookup table

## 输出契约

返回并落盘：

- trace path 或生成的 profile path
- framework
- model/server args，当可用时
- 模型结构图 Markdown，由 `model-architecture-diagram` 生成
- kernel table
- overlap-opportunity table
- fuse-pattern table
- 当精确匹配无法定论时，可选返回包含 `high` / `medium` / `low` 的 similarity note
- 一句简短总结，说明该 run 由什么主导
- overlap read 是来自 single-trace triage，还是 mapping/formal two-trace triage

文件产物要求：

- 指定工作目录下必须生成 `terminal_commands.log`
- 指定工作目录下必须生成最终 `.md` 报告，默认文件名为 `torch_profiler_analysis.md`
- 指定工作目录下应保留原始分析输出，默认文件名为 `analysis_stdout.txt`
- 如果分析脚本产生 stderr，保存为 `analysis_stderr.txt`


## 通用算子级 root-cause 归因（已下沉）

技能自带零依赖脚本 `torch_trace_scope_summary.py` 的算子级归因工作流、脚本输出说明与报告 checklist 移至 [references/operator-root-cause.md](references/operator-root-cause.md)。

