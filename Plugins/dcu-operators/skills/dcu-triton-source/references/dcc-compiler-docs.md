# DCC 编译器官方文档索引（dcu-triton-source / dcu-kernel-dev 共用参考）

> 来源：光合开发者社区文档中心 `/api-doc/*`，版本树 `ad998204-4e7c-11f1-8262-0242ac150003`（8 章 56 叶，2026-10-14 实测，未真机复测）。
> 取全文：`GET https://developer.sourcefind.cn/api-doc/document/TreeDocument?VersionUUID=<树uuid>` 列目录；`DetailDocument?UUID=<叶uuid>` 拿 md 全文（匿名）。
> z-file Cat12「DCC 编译器工具链」另有 dcc-autotuner / dcc-opentuner 独立工具包下载位；DTK 目录下还有 compiler/（DCC_25.01→26.06 独立版本流，11 版）与 dtk_extension_packages/（SYCL DPC++ 手册）。

## 八章结构（56 叶）
1 DCC 简介（GCVM IR 简介/在 DTK 工具链中的角色/语言与编程模型/工具链）
2 支持的异构编程模型（HIP：执行模型/语言特性/编译构建/调试性能分析；OpenMP：核心特性/指令速查/编译调试）
3 编译流程与编译选项（安装环境/统一编译流水线/Fatbinary 产出/**三个核心编译选项：--offload-arch、-O0~Ofast、默认参数注入**）
4 优化与调优（架构相关：LDS 向量化、LingShuUpdateAlign；架构无关：循环展开/LICM/LSR/SpeculativeLoopUnswitch；**专家级：GPU 执行与资源参数、浮点模型与精度控制、-mllvm 选项详解**；LTO）
5 工具链（工程链路/dcc 驱动与构建/**dcc-autotune 参数搜索（OpenTuner 算法）**/**dccprof 运行时分析**）
6 前端和 GCVM（多语言前端接入/GCVM 角色定位/**GCVM 接口与用法/AI 编译器接入 GCVM：Triton 接入、JAX/TensorFlow、IREE**）
7 LingShu 后端（定位/地址空间与存储层次/**SGPR/VGPR 分配与资源权衡**/**线程层级与硬件执行模型映射**/Code Object 结构/资源预算与寄存器调优/优化：atomic 接口统一、跨 lane 通信热点 atomic、Kernel 局部变量数组/lowering 与代码生成）
8 调试与诊断

## 与本库技能的对接
- `dcu-triton-source`：Triton hcu backend 走 aillvm；§6.4.1 Triton 接入 GCVM 是 Triton→DCC 链路的官方说明，排查 backend 行为差异时对照。
- `dcu-kernel-dev`：§7.3 SGPR/VGPR（DCU 768-vgprs 与 AMD CDNA 512 的差异根源在 LingShu 后端分配策略）、§4.3.3 -mllvm 选项可用于解释/调优 kernel 编译行为。
- `dcu-dtk-layout`：gcvm-link/llc/opt 等工具在 /opt/dtk 的实际落点对照 §5 工程链路。
- kernel 寄存器/占用率调优前先读 §7.3+§7.6（资源预算），别凭 NVIDIA 经验硬套。