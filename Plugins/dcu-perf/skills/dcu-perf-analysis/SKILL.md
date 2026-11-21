---
name: dcu-perf-analysis
description: DCU 性能分析全栈总纲：五层体系（框架/引擎/算子/kernel/硬件）与工具选型路由入口。触发场景：整模型或服务级性能问题、不知道该用哪个 profiler、要按层次拆解耗时。不适用：单个算子/kernel 的 PMC 读数与瓶颈归因走 ops 域的 op-profiling；算子卡的契约、bench 与合入判定走 op-contract / op-bench / op-gate。
version: 1.0.1
author: DCU Engineer
license: MIT
related_skills: [dcu-env, dcu-framework-profilers, dcu-perf-profiling, dcu-hipprof, dcu-rccl-test, dcu-op-optimization, dcu-perf-cases]
---

# DCU 性能分析全栈总纲

来源：《DCU性能分析及常用优化技术手册》（内部文档 file_id=<internal-file-id>，链接 <internal-doc-link>），含两个外链实战文档（LSTM-Unet-GNN 对比案例、rocblas/miopen size 抽取）。本 skill 是**方法论总纲 + 路由入口**，细节在各子 skill。

## 1. 五层性能分析体系

```
业务层（QPS、TPS、TTFT）
    ↓
框架层（PyTorch、TensorFlow、ORT、MIGraphX、vLLM、Megatron）
    ↓
Runtime层（ORT、TensorRT、MIGraphX）
    ↓
Kernel层（HIP / HIP Graph）
    ↓
硬件层（DCU、CPU、Memory、Network / Hylink-IB）
```

任一层都可能是瓶颈，分析时**自上而下逐层量化**，不要跳层下结论。

## 2. 标准分析步骤（强制顺序）

> 源自手册 Ch1「最佳实践分析步骤」，与 dcu-op-optimization 的闭环纪律一致。

1. **环境对齐（先排除软硬件差异）**：硬件/系统/驱动/VBIOS/DTK/AI 依赖库必须一致；否则任何 profile 结论不可复现。→ 见 `dcu-env` + 手册 Ch2。
2. **框架级 Profiler 定位热点**：用框架自带 Profiler（torch.profiler / tf.profiler / ORT / MIGraphX / vLLM trace / Megatron）拿到算子总耗时排序，确定瓶颈是**计算、数据搬运还是通信**。→ 见 `dcu-framework-profilers`（六大框架实战）。
3. **Kernel 级微架构分析**：对热点算子用 hipprof / rocprof 做 Kernel 级分析（SM 占用率、访存带宽、指令效率）；卷积类热点查 MIopen 日志。→ 见 `dcu-hipprof` / `dcu-perf-profiling`（含 MIopen 段）。
4. **分布式/服务级时间线排查**：多卡训练或推理服务用 hipprof / vLLM trace / Megatron profiling 看整体时间线，排查 CPU 调度空泡、IO、多 Stream 同步。→ 见 `dcu-hipprof` / `dcu-rccl-test`。

**每一步都必须量化对比、留痕、禁止"感觉快了"式结论**（详见 `dcu-op-optimization`）。

## 3. 工具选型路由

| 你要做的事 | 路由到 |
|---|---|
| 对齐环境 / 收集环境字段 / 锁频关 ACS | dcu-env（含手册 Ch2.1 字段表、dcuprofiletools 链接） |
| 框架级热点定位（PyTorch/TF/ORT/MIGraphX/vLLM/Megatron） | dcu-framework-profilers |
| Kernel 级微架构（hipprof 全集成：Trace/PMC/RCCL/泄漏/CoreDump） | dcu-hipprof（父，方法论总纲） |
| Kernel 级（rocprof/rocTracer 指标约定、MIopen 日志、优化方向 checklist、算子 size 抽取） | dcu-perf-profiling |
| 多卡 RCCL 通信 / MPI / xcl-lens 日志提取 / rccl-tests 构建 | dcu-rccl-test + dcu-hipprof |
| 改算子 / 验证正确性+性能闭环 | dcu-op-optimization |
| 看真实跨平台对标案例（LSTM/UNet/GNN 根因+优化前后数据） | dcu-perf-cases |

## 4. 常见优化方向（速记，详细见 dcu-perf-profiling）

- **计算-通信重叠**：Timeline 上 RCCL/NCCL 是否与 Kernel 真正并行；反向时通信应隐藏在计算后，否则调 bucket_size / 梯度累加。
- **Pipeline Bubbles**：Megatron PP 下气泡占比不应超过 `(pp_size-1)/num_microbatches`。
- **调度器延迟（vLLM）**：scheduler + Prefill/Decode 切换的显存整理（block copy）耗时 → 调 max_num_seqs / block_size；长则抬高 TTFT。
- **CPU 侧瓶颈**：Launch Overhead（hipExtModuleLaunchKernel 次数爆炸）→ 用 HIP Graph / CUDA Graph 回放；ncu/nsys 的 osrt 段查 GIL/DataLoader 阻塞。

## 4.1 微基准 → 端到端传导的三个判断（2026-08 LLM decode 实测方法论）

> 来源：Qwen3.8-27B w8a8 在 gfx936 4 卡 TP 的 decode 优化（A/B 序列）。**方法论可直接用**；文中具体百分比/耗时为该次实测，**仅参考**，换模型/硬件须复测。

- **微基准提速不线性传导到端到端**：kernel 微基准（纯 kernel 时间）快 N 倍 ≠ 端到端快 N 倍。decode 步里 GEMM 只占一部分（该例 ~35-47%），其余是 allreduce/elementwise/GDN/quant/attention/调度。把 GEMM 砍半端到端只 +6%；GEMM 再省 13% 端到端却 wash（淹没在 ±2-3% 基准噪声里）。**判断**：先算"该 kernel 占 decode 步的比例 × 微基准提速"得到端到端理论上限，上限 < 基准噪声（~±2-3%）就不值得做端到端 A/B，直接看微基准定案。
- **CUDA graph 只消 CPU 侧 launch 开销，不消 kernel 在 GPU 上的执行时间**：graph 把整步几百个 kernel 预录制、一次提交，压掉 kernel 间空隙（CPU 发射跟不上 GPU 的空转）；但 GEMM 的 FLOPs、allreduce 的字节数、quant 的显存读写是物理工作量，一分不少。**验证 GPU 是否打满**：profile 出的 kernel 时间总和/步 ≈ 端到端每 token 延迟 → GPU 基本打满，graph 收益已拿满，剩余空间只在 kernel 本身更快；两者差距大 → 还有 launch/调度空泡可挤。注意 EAGLE draft extend/采样/调度在 graph 外（trace 里可见 alloc/assign_req_to_token_pool 等零散调度 kernel）。
- **elementwise 碎片化判断**：decode 步的 elementwise 常高度碎片化（几十种小 kernel，单项 0.02-0.2ms 量级），没有可融合的"大对"时逐项 kernel 级工作收益 <1%，不值得做。**判断**：拆 elementwise 到单 kernel 粒度，最大单项 < 0.2ms/步 且无 2 个相邻可融合 kernel → 放弃，转下一个大类。
- **decode 步 kernel 分解方法（无 step 标注的 trace）**：用 allreduce kernel（如 `cross_device_reduce_1stage_pcie`）的间隔模式识别 decode 步——decode 步的 AR 与下一个 AR 间隔 1-3ms，prefill/draft 的 AR 密集打包 <0.5ms 间隔；据此把 kernel 归到 decode 步再分类（gemm/ar/elementwise/gdn/quant/attn/norm）。

## 5. DCU 纪律（沿用团队规范，必须带）

- 指定部分卡用 `HIP_VISIBLE_DEVICES=0,1,2,3`（编号从 0 起），不靠 `-g N` 自行挑卡。
- DCU 容器必挂 `-v /opt/hyhal:/opt/hyhal:ro`，否则 DCU 不可见（DEVCOUNT=0）。
- gfx 用物理机 `rocminfo` 查（带 ROCM_PATH/LD_LIBRARY_PATH，容器内不可靠）。
- 进容器跑测试前先查 `NCCL_NCHANNELS_PER_PEER` 是否已设（镜像常预置=2），有则 unset 再测。
- 编译/通测必须在**用户指定的专用容器**内做，不污染基础镜像。

## 6. 相关文档溯源

- 主手册：内部文档 file_id=<internal-file-id>（内部文档链接，已略）
- 实战案例（LSTM/UNet/GNN 跨平台对标）：file_id=<internal-file-id>（内部文档链接，已略）→ 见 dcu-perf-cases
- 算子 size 抽取 SOP（MIOPEN/ROCBLAS 日志 → 产品部优化）：file_id=<internal-file-id>（内部文档链接，已略）→ 见 dcu-perf-profiling
- 环境收集工具 dcuprofiletools：https://developer.sourcefind.cn/codes/tsoc/dcuprofiletools

## 7. 离线性能预估工具（AnalyzerModelTools）

> 两个**离线**性能预估工具（无需真跑硬件/模型，几秒出估算值）：`op_analyzer`（单算子）与 `model_analyzer`（整个 LLM 推理）。适合**标前测试、方案对比、并行策略预研、算子 shape 导出**。
> 参考：`references/analyzer-model-tools.md`（方法学：三种分析模式、10 个功能、支持算子/模型、常见使用模式速查、坑）；源码与 59 算子测试脚本在 `references/op_analyzer_test/` 与 `references/model_analyzer_py_src/`。
> ⚠ **本机 glibc 2.28 跑不了 C++ 后端**（报 GLIBC_2.29 not found）；需在 **Python 3.10 + glibc ≥ 2.29 的配套环境**（DCU 服务器）真实跑。wheel 是 cp310，python3.7.9 不匹配。

### op_analyzer — 单算子性能分析
- 三种模式：`compute`（已知实测耗时→算力/带宽）、`eval`（估算耗时）、**指定利用率**（按 TC/vector/SFU/带宽估算）。
- 59 个算子独立测试脚本，安装 wheel 后 `python3 test_<op>.py` 单跑。
- 数据类型枚举：`HIPDATA_FP32/FP16/BF16/I32/I64/I8/FP8E4M3/FP4E2M1/FP8E8M0`。

### model_analyzer — LLM 推理离线预估
- 10 个功能：单点性能（`model_analyzer()`）、利用率对比（`util_sweep`）、逐算子利用率（`util_profile`）、PD/超节点并行 sweep、最大 batch、算子 shape 导出、QPS 服务仿真。
- 20+ 模型别名、12 种硬件。`python -m model_analyzer.<模块>` 命令行可用。
- 常见使用模式速查表（文档 §速查） → 见 `references/analyzer-model-tools.md`。

### 与 DCU 测试场景衔接
- **标前测试**：离线预估目标模型 TPS/TTFT → 比选型/方案。
- **算子 benchmark 采集**：`tools.shape_collector` 生成 shape 清单 → DCU 服务器真实 benchmark → 回灌 dcu-perf-profiling / dcu-op-optimization。
- **并行策略预研**：`pd_sweep` / `pd_separate_sweep` 遍历优化组合 + 配平实例数。
- **QPS 服务能力**：`serving_sim` 扫 SLO 拐点。

估算数字是**离线预估，不是实测基线**——部署前仍须 DCU 真机验证。
