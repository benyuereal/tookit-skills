---
name: dcu-framework-profilers
description: 六大框架在DCU上Profiler实战：PT/TF/ORT/MIGraphX/vLLM/Megatron
version: 1.0.0
author: DCU Engineer
license: MIT
---

# 框架级 Profiler 实战（六大框架）

来源：《DCU性能分析及常用优化技术手册》Ch3（内部文档 file_id=<internal-file-id>）。框架层是性能分析五层体系中的**第二层**——在环境对齐（dcu-env）之后、Kernel 级（dcu-hipprof）之前，先用框架 Profiler 拿算子总耗时排序，确定瓶颈是计算/搬运/通信。

> DCU 上 PyTorch 基于 AMD ROCm 适配，`torch.cuda.*` 命名空间会自动映射到 HIP；TF/ORT/MIGraphX 同理。下文示例以 NVIDIA 命名书写，DCU 上按 ROCm 映射可直接复用。

## 框架 Profiler 选型总表

| 框架 | Profiler 工具 | 启用方式 | 关注指标 | 输出格式 |
|---|---|---|---|---|
| PyTorch | PyTorch Profiler | 代码内 `with profiler.profile()` | Kernel 耗时、显存分配、CPU-GPU 重叠、通信时间 | Chrome Trace JSON |
| TensorFlow | TF Profiler | `TensorBoard(profile_batch=...)` / `tf.profiler.experimental` | Step 时间、输入流水线延迟、XLA 效果、Op 统计 | TensorBoard 可视化 |
| ONNXRuntime | ORT 内置 Session Profiler | `sess_options.enable_profiling=True` | 节点级推理时间、内存占用、EP 选择效率 | JSON / CSV |
| MIGraphX | DCU Profiler + MIGRAPHX_TRACE_* | 环境变量 `MIGRAPHX_TRACE_COMPILE=1` 等 | 算子融合效果、HIP Kernel 耗时、内存拷贝、编译优化 | CSV / JSON / JSON-Trace |
| vLLM | PyTorch Profiler + VLLM profile | `VLLM_TORCH_PROFILER_DIR` / `--profile` | PagedAttention Kernel、调度延迟、KV Cache、解码阶段 | Chrome Trace JSON (Perfetto) |
| Megatron/LlamaFactory | PyTorch Profiler | `ENABLE_PROFILING=1` / `use_pytorch_profiler=True` | 通信-计算重叠、TP/PP/DP 效率、显存峰值、LoRA 效率 | Chrome Trace JSON |

## 1. PyTorch 框架分析

最通用的 Profiler，训练/VLLM/推理均可用，支持 CPU/CUDA/HIP 后端。

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA,   # DCU 上对应 HIP，ROCm 映射
    ],
    schedule=profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    on_trace_ready=profiler.tensorboard_trace_handler("./log"),
    record_shapes=True,        # 记录张量形状
    profile_memory=True,      # 记录显存分配
    with_stack=True,           # 记录调用栈
    with_flops=True,           # 估算 FLOPs
) as prof:
    for step, batch in enumerate(dataloader):
        train_step(batch)
        prof.step()
```

**关注点**
- Kernel 执行时间：识别耗时 Top 算子（如 `aten::conv2d`、`flash_attn`）。
- 显存分配模式：`profile_memory=True` 追踪 `cudaMalloc` 热点，发现显存碎片。
- CPU-GPU 重叠：检查 `cudaMemcpy` 是否与 Kernel 执行重叠，理想应完全隐藏。
- 通信瓶颈：分布式场景 `ncclAllReduce` 占比应 < 20% 时间。

## 2. TensorFlow 框架分析

与 TensorBoard 深度集成，适合输入流水线分析和 TPU 训练。

```python
# Keras 方式
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='./logs',
    profile_batch='500,510'  # 捕获第 500-510 个 batch
)
model.fit(..., callbacks=[tensorboard_callback])

# 底层 API 方式
tf.profiler.experimental.start('logdir')
# ... 训练代码 ...
tf.profiler.experimental.stop()
```

**关注点**
- Step Time 稳定性：波动大通常说明输入流水线或同步点问题。
- Idle Time：设备等待数据的空闲时间，理想应 < 5%。
- XLA 优化效果：`jit_compile=True` 前后 Op 融合对比。

## 3. ONNXRuntime 框架分析

跨框架中立执行引擎推理分析，支持 PyTorch/TF/Sklearn 导出模型。

```python
import onnxruntime as ort

sess_options = ort.SessionOptions()
sess_options.enable_profiling = True
sess_options.profile_file_prefix = "ort_profile"
sess_options.log_severity_level = 0  # Verbose

session = ort.InferenceSession(
    "model.onnx",
    sess_options,
    providers=['ROCMExecutionProvider', 'CPUExecutionProvider']  # DCU 走 ROCMExecutionProvider
)

outputs = session.run(None, {"input": input_data})
prof_file = session.end_profiling()  # 获取文件路径
```

脚本式性能分析：
```bash
onnxruntime_perf_test -m times -e rocm -r 20 -o 1 -I onnx_path
```

**关注点**
- 节点级耗时：识别 ONNX 图中瓶颈 Node（大 MatMul、Transpose）。
- EP 选择效率：对比 `CUDAExecutionProvider` vs `ROCMExecutionProvider` vs `CPUExecutionProvider`。
- 内存模式：`sess_options.enable_mem_pattern=True` 的内存复用效果。
- 线程配置：`intra_op_num_threads` 与硬件核心数匹配度。

## 4. MIGraphX 框架分析

DCU 专用推理图编译器，Profiler 依赖 DCU 工具链，关注编译时优化与运行时 HIP Kernel。

```python
# 编译阶段详细日志
export MIGRAPHX_TRACE_COMPILE=1
export MIGRAPHX_TRACE_EVAL=1
export MIGRAPHX_TRACE_BENCHMARKING=1

import migraphx
model = migraphx.parse_onnx("model.onnx")
model.compile(tgt="gpu")  # 编译日志输出优化 pass 信息
```

脚本式性能分析：
```bash
migraphx-driver perf [--enable-offload-copy] [-h] [--batch] [--input-dim] [-n] [--fp16] [--gpu/--cpu] [--nhwc] [--opt-compile] --onnx/--migraphx

# 参数说明：
# --enable-offload-copy：offload_copy 模式（输入在 device 端则不加）
# --batch：batchsize（onnx 未指定时需手动设）
# --input-dim：模型输入 shape
# -n：迭代次数（默认 100）
# --fp16：FP16 模式（默认 FP32）
# --gpu/--cpu：运行模式（默认 gpu）
# --onnx / --migraphx：模型文件路径
# -d：更详细数据
# --nhwc：nhwc 融合模式（仅 fp16）
# --opt-compile：auto-tune，对不同融合策略取最优编译
```

**关注点**
- 算子融合效果：`fuse_relu`、`fuse_conv_bn` 等 pass 是否生效。
- Gather 算子优化：多 Gather 水平融合，减少 Kernel 启动开销。
- HIP Kernel 耗时：hipprof 输出的 KernelName、GPUMinTime、GPUMaxTime（见 dcu-hipprof / dcu-perf-profiling）。
- 内存分配模式：MIGraphX EP 多 stream 内存管理。

## 5. vLLM 框架分析

大模型推理引擎，核心关注 PagedAttention、连续批处理、解码阶段效率。

```bash
# 启动设置环境变量
export VLLM_TORCH_PROFILER_DIR=./vllm_traces
export VLLM_TORCH_PROFILER_WITH_STACK=1
export VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY=1
export VLLM_RPC_TIMEOUT=1800000  # 防止 flush 超时

python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-70B \
    --tensor-parallel-size 4

# 触发 Profiling
curl -X POST http://localhost:8000/start_profile
curl -X POST http://localhost:8000/v1/completions -d '{"prompt":"Hello","max_tokens":50}'
curl -X POST http://localhost:8000/stop_profile
```

脚本式 / 离线推理：
```bash
vllm bench serve --model meta-llama/Llama-3.1-70B \
    --dataset-name sharegpt --num-prompts 2 --profile   # 自动触发 Profiler

# 离线
from vllm import LLM, SamplingParams
import os
os.environ["VLLM_TORCH_PROFILER_DIR"] = "./traces"
llm = LLM(model="meta-llama/Llama-2-7b")
llm.start_profile()
outputs = llm.generate("Hello world", SamplingParams(max_tokens=100))
llm.stop_profile()
```

**关注点**
- pagedAttention Kernel：`paged_attention_v1/v2_kernel` 耗时。
- Prefill vs Decode：Prefill（首 token）计算密集，Decode（后续）内存带宽密集。
- 调度开销：scheduler 和 prepare_input 不应占 Step 时间 >10%。
- KV Cache 管理：`copy_blocks`/`swap_blocks` 在 Prefix Cache 命中时耗时。

## 6. Megatron / LlamaFactory 训练框架分析

Megatron 面向 3D 并行（TP/PP/DP）；LlamaFactory 微调统一框架（SFT/LoRA/DPO/RLHF），Profiler 依赖底层 PyTorch，需手动注入。

```python
# 在 src/llamafactory/train/sft/workflow.py 或自定义训练循环中
import torch.profiler as profiler

def run_sft_profiler(model_args, data_args, training_args):
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
        schedule=profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
        on_trace_ready=profiler.tensorboard_trace_handler("./llama_profile"),
        record_shapes=True, profile_memory=True, with_stack=True,
    ) as prof:
        for step in range(training_args.num_train_epochs * len(trainer.get_train_dataloader())):
            trainer.train(resume_from_checkpoint=None)
            prof.step()
            if step >= 4:  # wait+warmup+active
                break
```

**关注点**
- LoRA 适配器开销：`lora_A`/`lora_B` 的 MatMul 是否成瓶颈（通常占比 <<5%）。
- 显存峰值：`profile_memory=True` 追踪 `max_memory_allocated`，验证 `per_device_train_batch_size`。
- 梯度累积影响：`gradient_accumulation_steps>1` 时前向占比上升。
- 数据预处理：template 和 cutoff_len 在 CPU 上的耗时。

## 7. 下一步路由

- 框架 Profiler 定位到热点算子后 → Kernel 级深挖：`dcu-hipprof`（Trace/PMC）或 `dcu-perf-profiling`（rocprof/MIopen）。
- 分布式通信瓶颈 → `dcu-hipprof` / `dcu-rccl-test`（含 xcl-lens）。
- 拿到优化方案要验证闭环 → `dcu-op-optimization`（强制正确性+性能量化对比）。
- 想看真实跨平台对标案例（根因+优化前后数据）→ `dcu-perf-cases`。

## 8. SGLang 实战（补充，未单列框架 Profiler 节）

SGLang 在 DCU 上的 Profiler 用法与 vLLM 类似（底层同走 PyTorch Profiler + Perfetto），但其**部署参数对性能影响极大**，常见坑不在 profiler 而在启动参数。典型实战：

- 多模态（含 mamba 结构，如 Qwen3.5/3.6）压测若并发被静默限流（设 32 只跑 20）→ 查 `max-mamba-cache-size` 缺省过小（详见 `dcu-vllm-deploy` 4.2 + `dcu-perf-cases` 案例 7 完整启动脚本模板）。
- 一堆 `SGLANG_USE_*` 环境变量（LIGHTOP/MARLIN/FUSED_TOPK_SOFTMAX/AITER_LINEAR_ATTN/EAGLE 投机等）是 BW 系多模态提速关键，模板见 `dcu-perf-cases` 案例 7。
- SGLang 海光适配源码与参数细节见 `dcu-sglang-source`（已实建为可用 skill）。

## 溯源

- 主手册 Ch3：内部文档 file_id=<internal-file-id>（内部文档链接，已略）
