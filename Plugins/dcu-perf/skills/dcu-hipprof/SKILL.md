---
name: dcu-hipprof
description: "hipprof性能分析总纲：Trace/PMC/RCCL/MPI/泄漏/CoreDump（子域已并入父，方法论总纲）。触发场景：要用 hipprof 采集一次 trace 或 PMC、分析通信/内存泄漏/core dump、弄清 hipprof 各子工具怎么跑。不适用：PMC 数字采回来之后怎么读、怎么判「卡在算力/内存/指令发射」、口径陷阱（TA_* 跨 SE、首次 launch 虚高）走 ops 域 op-profiling；先算上界/地板走 op-roofline。"
version: 1.0.1
author: DCU Engineer
license: MIT
---

# hipprof 性能分析最佳实践（dcu-hipprof）

hipprof 是 DTK 提供的 **HIP 应用性能分析工具**，可对应用程序的 API、Kernel、数据传输、通信、内存资源及硬件计数器进行采集和分析。本 skill 是父索引，覆盖方法论总纲；基线/Trace/PMC/RCCL/MPI/泄漏/CoreDump 等子域的方法论已并入本父 skill（见 §5）。

> 与 `dcu-perf-profiling` 的关系：后者基于 `rocprof`/`rocTracer`（更偏算子级指令采样），本 skill 基于 `hipprof`（更集成：Trace/PMC/RCCL/MPI/泄漏/CoreDump 一把抓）。二者互补，不冲突。

## 1. hipprof 简介

主要能力：
- HIP API、Kernel、Memcpy 时间线分析
- Perfetto 可视化 Trace 导出
- ROCTX/HIPTX 用户标记分析
- RCCL 通信分析
- MPI 多进程、多节点分析
- OpenMP/OpenACC、OpenCL、HSA、LIBC 接口分析
- PMC 硬件性能计数器分析
- 显存、内存、Stream、Event 泄漏检测
- 寄存器压力分析
- CoreDump 崩溃现场分析
- 集群性能数据收集与汇总

基本命令格式：
```shell
hipprof [options] <app command line>
```

## 2. 性能分析总体原则

### 2.1 分层定位，避免一次性开启全部功能
推荐"由粗到细"流程：
```
业务性能异常
    ↓
基础性能基线
    ↓
API / Kernel 统计分析
    ↓
Trace 时间线分析
    ↓
热点 Kernel 定位
    ↓
PMC 硬件计数器分析
    ↓
通信、访存、寄存器或代码级优化
    ↓
回归验证与结果固化
```
不建议一开始同时开启 Trace、PMC、LeakCheck 等全部功能——Profiler 本身有开销，可能影响业务真实性能。

### 2.2 区分性能瓶颈层级
<!-- sourceId: PLAKB4JIACAD4 (内部文档 hipprof 指南表) -->
| 层级 | 重点分析对象 | 常见问题 |
|---|---|---|
| 主机侧 | HIP API、CPU 调度、同步调用 | Kernel Launch 频繁、同步阻塞、Python 调度开销高 |
| 数据传输侧 | H2D、D2H、D2D、P2P Copy | 拷贝过多、带宽低、Copy 与计算未重叠 |
| 设备计算侧 | Kernel、LDS、L1/L2 Cache、ALU | 算力利用率低、访存受限、LDS Bank Conflict |
| 通信侧 | RCCL API、RCCL Kernel、MPI | AllReduce 耗时高、通信不均衡、跨节点带宽不足 |
| 资源稳定性 | 内存、显存、Stream、Event | 显存泄漏、内存波动、长期服务性能衰退 |

### 2.3 Trace 看流程，PMC 看原因
<!-- sourceId: KTAKB4JIADQDO (内部文档 hipprof 指南表) -->
| 工具能力 | 主要作用 |
|---|---|
| --hip-trace | 查看 HIP API、Memcpy、Kernel 的整体执行流程 |
| Perfetto Trace | 分析 Kernel、Copy、Stream 是否并行或存在空泡 |
| --pmc | 分析算力利用率、L1/L2、LDS 等硬件行为 |
| --pmc-read | 分析 L2 读请求、读数据量及缓存行为 |
| --pmc-write | 分析 L2 写请求、写数据量及写回压力 |
| --rccl-trace | 分析 AllReduce、AllGather、ReduceScatter 等通信行为 |
| --leak-check | 检测 Host、Device、Stream、Event 泄漏 |
| --codeobj-analyze | 分析 Kernel 寄存器压力 |
| --input-core | 分析 Kernel Fault、AQL 异常和 CoreDump 现场 |

## 3. 推荐性能分析闭环（全篇总纲）
```
明确性能问题
   └─ 吞吐低、TTFT 高、通信慢、显存增长、Kernel Fault
建立无 Profiler 基线
   └─ 记录吞吐、时延、显存、利用率、通信带宽
统计定位
   └─ hipprof --hip-trace --stats
时间线确认
   └─ 分析 API、Copy、Kernel、RCCL、Stream
热点深入
   └─ PMC、RCCL Trace、ROCTX、寄存器压力
实施优化
   └─ 算子融合、异步化、Batch 优化、访存优化、通信优化、显存复用
回归验证
   └─ 对比吞吐、时延、P99、TTFT、显存、带宽
固化最佳实践
   └─ 保存命令、Trace、CSV、环境信息、优化结论
```

## 4. 最终建议：三层工具链
```
第一层：Trace 分析  → 定位 API、Memcpy、Kernel、RCCL 时间线问题
第二层：PMC 分析    → 定位 ALU、LDS、L1、L2、访存与缓存问题
第三层：稳定性分析  → 定位内存泄漏、寄存器压力、CoreDump 和 Kernel Fault
```
推荐执行顺序：
```shell
# 第一轮：快速确定热点
hipprof --hip-trace --stats ./your_app
# 第二轮：确认时间线、空泡及并行关系
hipprof --hip-trace --group-stream ./your_app
# 第三轮：深挖热点 Kernel
hipprof --pmc --kernel-name "hot_kernel" ./your_app
# 第四轮：多卡通信专项分析
hipprof --hip-trace --rccl-trace --show-pid ./your_app
```

## 5. 方法论已并入父 skill（子目录已清理）

> **合并策略（对应只读分析结论）**：六个子 skill 的方法论内容（基线/Trace/PMC/通信/数据/稳定性）高度重叠，现**已收敛并入本父 skill**。子目录（dcu-hipprof/pmc/rccl-mpi/stability/data-control）已物理删除，不再构成独立 skill。保留物已转述进父 skill 的 §5 方法闭环 + §6 命令模板区。

| 原子目录 | 方法论已并入父 skill 的章节 |
|---|---|
| baseline-trace（基线/统计/Trace/ROCTX） | §2 总体原则 + §6 命令模板 |
| pmc（PMC 硬件计数器/L2 读写） | §2.3 Trace 看流程/PMC 看原因 + §6 |
| rccl-mpi（RCCL 通信/MPI） | §6 + §7 使用须知 |
| data-control（长时 Session/数据量控制） | §6 |
| stability（泄漏/寄存器/CoreDump） | §6 |

> 注：物理删除后，原子 skill 专属脚本/前置纪律现已在父 skill 的「§6 常用命令模板速查」区直接给出（含 `-v /opt/hyhal:/opt/hyhal:ro`、`HIP_VISIBLE_DEVICES`、`NCCL_NCHANNELS_PER_PEER` 等）。
> 覆盖范围：父 skill 已含全部方法论；不再需要子 skill 定向入口——agent 直接按 §5 闭环/§6 命令模板操作即可。

## 6. 常用命令模板速查
```shell
# 单卡基础分析
hipprof --hip-trace --stats -o profile_single ./your_app
# 单卡完整时间线
hipprof --hip-trace --group-stream -o trace_single ./your_app
# 多卡通信分析
hipprof --hip-trace --rccl-trace --show-pid -o trace_rccl ./your_app
# 热点 Kernel PMC 分析
hipprof --pmc --kernel-name "your_hot_kernel" --pmc-type 3 -o pmc_hot ./your_app
# 内存泄漏分析
hipprof --leak-check --show-limit 30 --record-memfreeinfo -o leak_profile ./your_app
# 长时间服务问题窗口采集
hipprof --session server_profile --hip-trace --flush-interval 1000 --exit-cleanup ./your_server
# MPI 多节点性能采集
mpirun -np 8 -H node1:4,node2:4 hipprof --hip-trace --mpi-trace --show-pid ./your_mpi_app
```

## 7. 使用须知
- 经验仅供参考。先核对**设备型号 + 驱动/DTK 版本 + 报错原文**再下手。
- profiling 数字（吞吐/带宽/占比等）**严禁写入本 skill**（会变/过期），只存方法、命令、判定框架。
- 高危操作谨慎；优先测试机验证，确认带外恢复通道。
- 容器内跑 profiling 的通用纪律见各子 skill 前置说明（含 `-v /opt/hyhal:/opt/hyhal:ro`、`HIP_VISIBLE_DEVICES`、`NCCL_NCHANNELS_PER_PEER` 等）。
