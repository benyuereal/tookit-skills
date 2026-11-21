# dcu-perf-analysis（DCU 性能分析全栈总纲）

用户向 README。

## 是什么
性能分析的**方法论总纲 + 路由入口**，不绑定具体工具。把《DCU性能分析及常用优化技术手册》的核心框架抽出来：五层体系 + 标准分析步骤 + 工具选型路由。当用户要做性能分析/优化、不确定用哪个工具、或要"先量化再改"时加载。

## 核心内容
- **五层分析体系**：业务层 → 框架层 → Runtime 层 → Kernel 层 → 硬件层（DCU/CPU/内存/网络）。任一层都可能瓶颈，分析时自上而下逐层量化，不要跳层下结论。
- **标准步骤（强制顺序）**：
  1. 环境对齐（先排除软硬件差异）→ `dcu-env`
  2. 框架级 Profiler 定位热点（算子总耗时排序）→ `dcu-framework-profilers`
  3. Kernel 级微架构分析（SM 占用/带宽/指令效率，卷积查 MIopen）→ `dcu-hipprof` / `dcu-perf-profiling`
  4. 分布式/服务级时间线排查（多卡 training / 推理服务 CPU 空泡、IO、多 Stream 同步）→ `dcu-hipprof` / `dcu-rccl-test`
  - 每一步都必须量化对比、留痕，禁止"感觉快了"式结论。
- **工具选型路由表**（你要做什么 → 路由到哪个 skill）。
- **常见优化方向速记**：计算-通信重叠、Pipeline Bubbles、调度器延迟（vLLM）、CPU 侧 Launch Overhead（HIP Graph 回放）。
- **DCU 纪律**：`HIP_VISIBLE_DEVICES=0,1,2,3`（编号从 0 起）；容器必挂 `-v /opt/hyhal:/opt/hyhal:ro`；gfx 用物理机 `rocminfo`；进容器先查 `NCCL_NCHANNELS_PER_PEER`（镜像常预置=2，有则 unset）；通测在用户指定专用容器内做。
- **离线性能预估工具（AnalyzerModelTools）**：`op_analyzer`（单算子）+ `model_analyzer`（LLM 推理），离线估算 eval_time/tflops/TPS/TTFT。参考在 `references/analyzer-model-tools.md` + `references/op_analyzer_test/` + `references/model_analyzer_py_src/`；⚠ 本机 glibc 2.28 跑不了 C++ 后端，需 Python 3.10 + glibc≥2.29 配套环境。

## 相关 skill
dcu-env / dcu-framework-profilers / dcu-perf-profiling / dcu-hipprof / dcu-rccl-test / dcu-op-optimization / dcu-perf-cases

## 溯源
主手册 内部文档 file_id=<internal-file-id>（《DCU性能分析及常用优化技术手册》）；实战案例见 `dcu-perf-cases`；算子 size 抽取 SOP 见 `dcu-perf-profiling`；环境收集工具 dcuprofiletools：https://developer.sourcefind.cn/codes/tsoc/dcuprofiletools
