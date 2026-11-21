---
name: dcu-rccl-source
description: DTK RCCL源码级分析：PCIe混合链路优化
version: 1.0.1
author: DCU Engineer
license: MIT
---

# 海光 DTK RCCL 源码分析：PCIe 混合链路优化（dcu-rccl-source）

适用场景：理解/调优海光 DTK RCCL 的 **PCIe 混合链路**（在 xHCL + PCIe 并存拓扑下，如何通过环境变量控制 channel 数量与缓冲区放置，把通信 traffic 引导到 PCIe link memory 路径）。本文是**源码级地图**，回答"代码在哪、为什么这么设计、怎么调"。

> **来源**：内部文档《深入理解 DTK RCCL 的 PCIe 混合链路优化》（file_id `<internal-file-id>`，DTK 26.04，20260325，纯自包含已全文抓取）。
> ⚠ **源码纪律**：本文源码片段基于 **DTK 26.04.2 的 custom-rccl（海光 RCCL fork）**，行号（如 `src/init.cc:1638`、`src/transport/p2p.cc:119-120`、`src/transport/p2p.cc:996-1010`）**随版本可能漂移**，但本 skill 新增了 **`references/dtk-26.04.2-rccl-source-map.md`**（真实源码树地图 + RCCL 专属环境变量总表 + PCIe 混合链路关键代码真实行号），实际改代码/查逻辑时**优先以该 reference 的行号核验**。行号漂移时，用 `grep -n` 在目标分支源码中实时定位。与 `dcu-vllm-source-*` 同源分析纪律一致。**源码树未随本包分发**（上游代码合规红线见 `docs/STANDARDS.md`）；核行号前请先自行获取 DTK 26.04.2 的 RCCL 源码（custom-rccl）。
> ⚠ **不存性能数字**：本文是机制解析，无基准/逐行日志；调优效果请用 `dcu-rccl-test` 实测。

## 0. 一句话机制

PCIe 混合链路优化 = **限制非 PCIe 路径的 channel 数量** + 控制通信 buffer 的放置策略（普通 shareable buffer vs 专用 PCIe link memory）。三个环境变量协同完成这件事。

## 1. 三个核心概念（源码里的基础抽象）

DTK RCCL 中 PCIe 混合链路优化实际调节的是：在既定拓扑和传输机制下，**如何选择不同 channel 的缓冲区放置策略**。三个基础概念：

### 1.1 互联拓扑 linkTopoType（底层硬件拓扑）
| 类型 | 代码枚举 | 含义 |
|---|---|---|
| xHCL 直连 | `HCU_LINK_MODE_XHCL_DIRECT` | GPU 之间存在 xHCL 直连链路 |
| xHCL 交换式互联 | `HCU_LINK_MODE_XHCL_HYSWITCH` | GPU 通过 HYSwitch 互联 |
| 纯 PCIe | `HCU_LINK_MODE_ONLY_PCIE` | 节点内没有 xHCL，只能走 PCIe |

### 1.2 数据传输 p2pType（P2P 连接如何建立/访问）
| 类型 | 代码枚举 | 含义 |
|---|---|---|
| Direct | `P2P_DIRECT` | 同进程 GPU 之间直接通过指针访问 |
| Intermediate | `P2P_INTERMEDIATE` | 通过中间 rank 转发 |
| IPC | `P2P_IPC` | 通过 IPC 机制共享缓冲区 |
| CUMEM | `P2P_CUMEM` | 通过 cuMem / shareable buffer 机制共享 |

### 1.3 缓冲区放置策略 p2pMemType（承载通信数据的 buffer 放哪类内存）
| 类型 | 代码枚举 | 含义 |
|---|---|---|
| 普通 P2P 共享内存路径 | `P2P_HIP_NOR_MEM` | 常规 shareable buffer 分配路径（普通 HIP/IPC） |
| PCIe link memory 路径 | `P2P_HSA_PCIE_MEM` | 用 `rocmAllocPcieLinkMem()` 分配的 PCIe P2P 内存，专用 PCIe 扩展路径 |

## 2. Collective vs P2P 路径（不是 API 名，是调度路径）

| 概念 | 任务类型 | 调度入口 | 默认 connIndex |
|---|---|---|---|
| Collective | `ncclTaskColl` → `scheduleCollTasksToPlan()` | 0 |
| P2P | `ncclTaskP2p` → `scheduleP2pTasksToPlan()` | 1 |

`connIndex` 是同一 channel、同一对端点之间的连接槽位索引（0/1/2），选 channel 上该端点对的一组连接上下文（connector/connInfo/proxy/buffer 状态）。默认 collective=connIndex 0、P2P=connIndex 1；满足条件可切到 connIndex 2（`NCCL_CONN_IDX_P2P_NET`）。

**走哪条路径的操作总结**：
- 明确 collective：AllReduce, AllGather, ReduceScatter, Broadcast, Reduce, AlltoAllPivot
- 明确 P2P：Send, Recv, AlltoAllv, Gather, Scatter
- 条件分流：AlltoAll 满足 pivot → AlltoAllPivot（collective 路径），否则展开为一组 Send/Recv（P2P 路径）

## 3. 三个关键环境变量（本文核心）

| 环境变量 | 生效条件 | 作用 |
|---|---|---|
| `NCCL_SIMPLE_CHANNELS` | gfx936（BW150/BW1000）和多节点 gfx938（BW1100）；默认 Simple 协议 channels 上限 16 | 控制 collective 路径在 Simple 协议下可使用的 channel 窗口，影响这些 channel 的 Simple buffer 分配范围 |
| `RCCL_COLL_XHCL_CHANNEL_NUM` | 单节点且有 xHCL 互联，否则改用 `MAXCHANNELS` | 控制 collective 路径中保留普通 P2P/xHCL buffer 放置的 channel 数量阈值；超出阈值的 channel 切到 PCIe link memory |
| `RCCL_P2P_XHCL_CHANNEL_NUM` | 单节点且有 xHCL 互联，否则改用 `MAXCHANNELS` | 控制 P2P 路径中保留普通 P2P/xHCL buffer 放置的 channel 数量阈值；超出阈值的 channel 切到 PCIe link memory |

**协同关系**：
- `NCCL_SIMPLE_CHANNELS` 限制 collective 的 Simple channel 数量
- `RCCL_COLL_XHCL_CHANNEL_NUM` 作用于 collective 路径，超出数量的 channels 改走 PCIe link memory
- `RCCL_P2P_XHCL_CHANNEL_NUM` 作用于 P2P 路径，超出数量的 channels 改走 PCIe link memory

**设置示例（形如，具体取值按拓扑/调优目标，本文未给推荐值，不编造）**：
```bash
export NCCL_SIMPLE_CHANNELS=16          # gfx936/BW150 等生效；-2 表示用默认逻辑
export RCCL_COLL_XHCL_CHANNEL_NUM=8     # collective 路径保留前 N 个 channel 走普通/xHCL buffer
export RCCL_P2P_XHCL_CHANNEL_NUM=8      # P2P 路径保留前 N 个 channel 走普通/xHCL buffer
# 超出阈值的 channel 自动切到 PCIe link memory (P2P_HSA_PCIE_MEM)
```

## 4. 源码级定位（高价值，基于 DTK 26.04 custom-rccl）

### 4.1 NCCL_SIMPLE_CHANNELS
- 代码位置：`src/init.cc`、`src/enqueue.cc`、`src/transport/p2p.cc`
- 变量定义：
```c
NCCL_PARAM(CollSimpleChannels, "SIMPLE_CHANNELS", -2)
```
- 作用：限制 collective 在 Simple 协议下使用的 channel 数量；限制 P2P transport 中哪些 collective channels 会为 Simple 协议分配完整缓冲区。
- 初始化逻辑（真实行号 `custom-rccl/src/init.cc:1638`，与 reference 一致；主文档原 `1672-1679` 为 26.04 内部文档提取版，已漂移）：
```c
  comm->CollSimpleChannels = comm->nChannels;
  // set Simple nChannels to ncclParamCollSimpleChannels() when gcn=gfx936 or multi nodes on gfx938
  if(IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx936") ||
    ((IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx938")) && comm->nNodes > 1)) {
    int CollSimpleChannels = 16;
    if(ncclParamCollSimpleChannels() != -2) CollSimpleChannels = ncclParamCollSimpleChannels();
    if(CollSimpleChannels <= comm->nChannels && CollSimpleChannels > 0) comm->CollSimpleChannels = CollSimpleChannels;
  }
```
- 结论：该变量**目前只在 gfx936（BW150/BW1000）和多节点 gfx938（BW1100）起作用**；默认 Simple 协议 channel 数量目标上限 16，实际生效值 `min(comm->nChannels, 16)`。

### 4.2 RCCL_COLL_XHCL_CHANNEL_NUM
- 代码位置：`src/transport/p2p.cc`
- 变量定义：
```c
RCCL_PARAM(CollXHCLChannelNum, "COLL_XHCL_CHANNEL_NUM", MAXCHANNELS)
```
- 作用：限制 collective 路径中保留普通 P2P/xHCL buffer 放置的 channel 数量；超出阈值的 collective channels 切到 PCIe link memory。
- 关键逻辑：
```c
static int getCollXhclChannelNum(struct ncclComm* comm) {
  return (comm->nNodes > 1 || comm->linkTopoType == HCU_LINK_MODE_ONLY_PCIE) ? MAXCHANNELS : rcclParamCollXHCLChannelNum();
}
```
- 阈值切换（`channelId` 越界则切 PCIe link memory）：
```c
if (checkXhclMemChannel(...)) {
  p2pBuff->memType = P2P_HIP_NOR_MEM;
} else {
  NCCLCHECK(rocmAllocPcieLinkMem(...));
  p2pBuff->memType = P2P_HSA_PCIE_MEM;
}
```

### 4.3 RCCL_P2P_XHCL_CHANNEL_NUM
- 代码位置：`src/transport/p2p.cc`
- 变量定义：
```c
RCCL_PARAM(P2pXHCLChannelNum, "P2P_XHCL_CHANNEL_NUM", MAXCHANNELS)
```
- 作用：限制 P2P 路径中保留普通 P2P/xHCL buffer 放置的 channel 数量；超出阈值的 P2P channels 切到 PCIe link memory。
- 关键逻辑：
```c
static int getP2pXhclChannelNum(struct ncclComm* comm) {
  return (comm->nNodes > 1 || comm->linkTopoType == HCU_LINK_MODE_ONLY_PCIE) ? MAXCHANNELS : rcclParamP2pXhclChannelNum();
}
```
- `checkXhclMemChannel()` 根据 `connIndex` 选择用 COLL 还是 P2P 阈值：
```c
if (connIndex == 0) {
  return channelId < collXhclChannelNum;
} else if (connIndex == 1) {
  return channelId < p2pXhclChannelNum;
}
```

## 5. 调优指向（与 dcu-rccl-test 衔接）

- **适用场景**：单节点且有 xHCL 互联（HCU_LINK_MODE_XHCL_DIRECT / HYSWITCH）时，用 `*_XHCL_CHANNEL_NUM` 限制保留在普通/xHCL buffer 的 channel 数，把超额 channel 逼到 PCIe link memory 路径，做混合链路权衡。
- **多节点或纯 PCIe 拓扑**：`*_XHCL_CHANNEL_NUM` 自动退化为 `MAXCHANNELS`（即不切 PCIe link memory），此时调它无效。
- **gfx936 / 多节点 gfx938 之外**：`NCCL_SIMPLE_CHANNELS` 不生效（默认逻辑不走 16 上限分支）。
- **实测验证**：调完用 `dcu-rccl-test` 跑 allreduce / alltoall 带宽与延迟，确认链路选择符合预期；本文不提供基准数字。

## 6. 与兄弟 skill 关系
- `dcu-rccl-test`：RCCL 测试/多卡带宽实测（本文机制调优后用它验证）。
- `dcu-topology`：互联拓扑（xHCL/HYSwitch/PCIe）采集，与本文 linkTopoType 对应。
- `dcu-vllm-source/0.21.0/0.23.0`：同范式的"源码结构 + 适配点"索引（vLLM 侧）。

## 使用须知
- 本文源码片段基于 **DTK 26.04.2 custom-rccl**，行号可能随版本漂移；实际改代码拉海光 fork 实时核验。
- DTK-26.04.2 的完整 RCCL 源码树（custom-rccl：src/ 源码主体 + CMakeLists/Makefile/build_ut.sh 等构建文件，含 msccl 算法 XML 与拓扑测试数据）**未随本包分发**（上游代码合规红线见 `docs/STANDARDS.md`）；需引源码原文/编译时，请先自行获取 DTK 26.04.2 的 RCCL 源码。
- 搭配 **references/dtk-26.04.2-rccl-source-map.md**：该地图给出 26.04.2 真实源码树 + RCCL 环境变量总表 + PCIe 混合链路关键代码行号，改代码/查逻辑优先用它核验。
- 搭配 **references/build_ut.sh-compile-path.md**：在 DCU 上编译 RCCL 动态库/UT 用例时，核对 hipcc/ROCM 路径与 BUILD_TESTS=ON 是否命中。
- 不存性能基线/逐行日志；调优效果用 `dcu-rccl-test` 实测留痕。
- 环境变量具体取值按拓扑与调优目标定，本文不给推荐值（不编造）。
- 高危操作谨慎，优先测试机验证。
