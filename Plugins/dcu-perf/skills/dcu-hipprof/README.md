# dcu-hipprof — 海光 DCU hipprof 集成剖析器最佳实践

hipprof 是 DTK 提供的 **HIP 应用性能分析工具**，可对 API、Kernel、数据传输、通信、内存资源及硬件计数器进行采集分析（Trace / PMC / RCCL / MPI / 泄漏 / CoreDump 一把抓）。本目录是**父 skill**，负责方法论总纲；基线/Trace/PMC/RCCL/MPI/泄漏/CoreDump 等子域方法论已并入本父，不再有独立子 skill。

> 与 `dcu-perf-profiling`（rocprof/rocTracer 系）互补：本 skill 用 hipprof 做集成剖析，后者更偏算子级指令采样。

## 三层工具链

```
第一层：Trace 分析  → API / Memcpy / Kernel / RCCL 时间线问题
第二层：PMC 分析    → ALU / LDS / L1 / L2 / 访存与缓存问题
第三层：稳定性分析  → 内存泄漏 / 寄存器压力 / CoreDump / Kernel Fault
```

## 目录

```
dcu-hipprof/
├── SKILL.md                       父：简介/原则/闭环/三层工具链/方法论总纲/命令速查
└── README.md                      本文件
```

## 内容覆盖（子域方法论已并入父）

> 基线/Trace/PMC/RCCL/MPI/泄漏/CoreDump 等子域方法论已收敛并入本父 skill（见 SKILL.md §5 合并映射表 + §6 命令模板），不再有独立子 skill 目录。

## 快速用法

```shell
# 第一轮：快速确定热点
hipprof --hip-trace --stats ./your_app
# 第二轮：确认时间线、空泡及并行关系
hipprof --hip-trace --group-stream ./your_app
# 第三轮：深挖热点 Kernel
hipprof --pmc --kernel-name "hot_kernel" ./your_app
# 第四轮：多卡通信专项
hipprof --hip-trace --rccl-trace --show-pid ./your_app
```

详见各子 skill 的 `SKILL.md`。
