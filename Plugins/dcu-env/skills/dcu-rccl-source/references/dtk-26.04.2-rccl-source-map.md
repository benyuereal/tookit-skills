# DTK 26.04.2 custom-rccl 源码树地图（dcu-rccl-source 增量参考）

> 本文件是对 `/root/custom-rccl-dtk-26.04.2.zip` 完整源码的**源码级地图**，与 `dcu-rccl-source` SKILL.md 主体（内部文档提取的机制解析版）互补。用于实际改代码/查逻辑时实时核验行号。
> 纪律：不存性能数字、不存逐行日志；仅沉淀源码结构、关键文件定位、环境变量真实定义位置、关键代码行号（源码纪律与 `dcu-vllm-source-*` 一致）。

## 0. 源码树总览（26.04.2 分支）

顶层目录结构：

```
custom-rccl-dtk-26.04.2/
├── .azuredevops/          # CI 配置（nightly/PR/slurm 多节点）
├── .github/              # CI 配置（therock CI，workflows）
├── cmake/                # CMake 逻辑（Dependencies, DownloadProject, MSCCLPP）
├── docker/               # Dockerfile.ubuntu 等
├── docs/                 # api-reference, doxygen, data/how-to
├── ext-net/ ext-profiler/ ext-src/ ext-tuner/   # 扩展：google-fastsocket、patches、tuner demo
├── makefiles/            # Makefile 逻辑（src/Makefile 等）
├── pkg/                  # 打包（deb/rpm）
├── src/                  # 核心源码（RCCL 主实现）
├── test/                 # Googletest 单元测试
├── tools/                # 辅助工具
├── CHANGELOG.md, LICENSE.txt, NOTICES.txt, README.md
```

`src/` 核心源码目录：

```
src/
├── allocator.cc bootstrap.cc channel.cc collectives.cc debug.cc enqueue.cc enhcompat.cc
├── group.cc init.cc init_nvtx.cc mnnvl.cc msccl.cc nccl.h.in nccl.pc.in proxy.cc
├── rccl_wrap.cc symmetric.cc transport.cc
├── device/             # 设备抽象（common.h 等）
├── graph/              # 拓扑图（connect.cc paths.cc search.cc rome_models.cc xml.cc topo.cc）
├── include/            # 头文件（param.h rccl_vars.h nccl_common.h p2p.h transport.h 等）
├── misc/               # 杂项（npkit, profiler, recorder, register 等）
├── ras/                # RAS 相关
├── transport/          # 传输层（p2p.cc coll_net.cc net.cc net_ib.cc net_socket.cc nvls.cc shm.cc 等）
├── plugin/             # 插件扩展
└── register/           # 注册表
```

## 1. RCCL 专属环境变量总表（真实源码定义点）

`param.h` 定义两种宏：`NCCL_PARAM(name, env, deftVal)`（前缀 `NCCL_`）与 `RCCL_PARAM(name, env, deftVal)`（前缀 `RCCL_`）。RCCL 专属环境变量分布在 `src/init.cc`、`src/enqueue.cc`、`src/graph/*.cc`、`src/transport/p2p.cc`。

| 环境变量（RCCL_ 前缀） | 真实源码定义位置 | 默认值 | 作用摘要 |
|---|---|---|---|
| `DEBUG_SLEEP` | `src/init.cc:106` | 0 | 调试挂起等待 attach |
| `FORCE_DETECT_TOPO` | `src/init.cc:136` | 0 | 强制探测拓扑（覆盖默认） |
| `GFX9_CHEAP_FENCE_OFF` | `src/init.cc:135` | 0 | 便宜 fence 关闭（gfx9） |
| `MSCCLPP_ENABLE` | `src/init.cc:132` | defaultEnableMscclpp | MSCCL++ 开关 |
| `MSCCLPP_FORCE_ENABLE` | `src/init.cc:133` | 0 | MSCCL++ 强制开启 |
| `MSCCLPP_THRESHOLD` | `src/init.cc:128` | 16MB | MSCCL++ 阈值 |
| `ENABLE_CONTEXT_TRACKING` | `src/init.cc:702` | 0 | 上下文跟踪 |
| `ENABLE_PROXY_TRACE` | `src/init.cc:698` | 0 | proxy 跟踪 |
| `KERNEL_COLL_TRACE_ENABLE` | `src/init.cc:700` | 0 | 内核集合跟踪 |
| `KERNEL_COLL_TRACE_THREAD_ENABLE` | `src/init.cc:701` | 0 | 内核集合跟踪线程 |
| `LL128_FORCE_ENABLE` | `src/init.cc:1011` | 0 | LL128 协议强制开启 |
| `P2P_NET_DISABLE` | `src/init.cc:1009` | 1 | P2P 网络禁用 |
| `PIVOT_ALLTOALL_ENABLE` | `src/init.cc:1010` | 1 | AlltoAll Pivot 开关 |
| `SHARED_NET_ENABLE` | `src/init.cc:1637` | 0 | 共享网络 |
| `STACK_SIZE_OVERRIDE` | `src/init.cc:2478` | 0 | 栈大小覆盖 |
| `INJECT_FAULTS` | `src/init.cc:1062` | 0 | 故障注入（测试用） |
| `NET_GDR_WRITE_USE_XDP_ENABLE` | `src/init.cc:1532` | 1 | GDR 写用 XDP 开关 |
| `GDR_WRITE_USE_XDP_ENABLE` | `src/init.cc:1532` | 1 | （同上，别名） |
| `COLL_SIMPLE_CHANNELS` | `src/init.cc:1638`（实际 `RCCL_PARAM(CollSimpleChannels, "SIMPLE_CHANNELS", -2)`） | -2 | Simple 协议 channel 数量上限（gfx936/多节点 gfx938） |
| `INTRANET_THRESHOLD` | `src/enqueue.cc:658` | 8388608 | 内网阈值 |
| `P2P_BATCH_ENABLE` | `src/enqueue.cc:961` | 0（64k） | P2P 批处理开关 |
| `P2P_BATCH_THRESHOLD` | `src/enqueue.cc:955` | 1<<16=64k | P2P 批处理阈值 |
| `P2P_NET_THRESHOLD` | `src/enqueue.cc:945` | 131072 | P2P 网络阈值 |
| `DISABLE_RAIL_TREES` | `src/graph/rome_models.cc:1907` | 0 | 禁用 rail tree 建模 |
| `MODEL_REVERSAL_DISABLE` | `src/graph/rome_models.cc:1267` | 0 | 模型反转禁用 |
| `OUTPUT_TREES` | `src/graph/connect.cc:711` | 0 | 输出树结构 |
| `ENABLE_INTRANET` | `src/graph/paths.cc:828` | -2 | 内网开关 |
| `MODEL_MATCHING_DISABLE` | `src/graph/search.cc:1043` | 0 | 模型匹配禁用 |

> `CollSimpleChannels` 定义在 `src/init.cc:1638`（`NCCL_PARAM` 前缀实为 `NCCL_SIMPLE_CHANNELS`），但实际生效逻辑在 `src/init.cc` 的 `if(IsArchMatch...gfx936...)` 分支（见 SKILL.md 主文档 4.1）。
> `COLL_XHCL_CHANNEL_NUM` / `P2P_XHCL_CHANNEL_NUM` 定义在 `src/transport/p2p.cc:119/120`（`RCCL_PARAM` 前缀）。

## 2. PCIe 混合链路关键代码定位（真实行号）

`src/transport/p2p.cc` 是 PCIe 混合链路的核心实现文件。

### 2.1 p2pMemType 枚举 + 结构定义

- `src/transport/p2p.cc:24-25`：`P2P_HIP_NOR_MEM`（普通 shareable buffer）/ `P2P_HSA_PCIE_MEM`（PCIe link memory）枚举
- `src/transport/p2p.cc:27-36`：`struct ncclP2pBuff`（directPtr/size/ipcDesc/memType）
- `src/transport/p2p.cc:38-46`：`struct p2pConnectInfo` / `p2pShmProxyInfo`（描述连接/共享内存代理）

### 2.2 环境变量定义行

- `src/transport/p2p.cc:119`：`RCCL_PARAM(CollXHCLChannelNum, "COLL_XHCL_CHANNEL_NUM", MAXCHANNELS)`
- `src/transport/p2p.cc:120`：`RCCL_PARAM(P2pXHCLChannelNum, "P2P_XHCL_CHANNEL_NUM", MAXCHANNELS)`

### 2.3 阈值函数（getCollXhclChannelNum / getP2pXhclChannelNum）

- `src/transport/p2p.cc:122-127`：
```c
static int getCollXhclChannelNum(struct ncclComm* comm) {
  // Do not use extend pcie mem channel for mult nodes and only pcie link
  return (comm->nNodes > 1 || comm->linkTopoType == HCU_LINK_MODE_ONLY_PCIE) ? MAXCHANNELS : rcclParamCollXHCLChannelNum();
}
static int getP2pXhclChannelNum(struct ncclComm* comm) {
  return (comm->nNodes > 1 || comm->linkTopoType == HCU_LINK_MODE_ONLY_PCIE) ? MAXCHANNELS : rcclParamP2pXHCLChannelNum();
}
```
- 语义：多节点或纯 PCIe 拓扑 → 不切 PCIe link memory（退化为 MAXCHANNELS）；单节点且有 xHCL 互联 → 按用户参数限制保留在普通/xHCL buffer 的 channel 数。

### 2.4 checkXhclMemChannel（connIndex 分流）

- `src/transport/p2p.cc:132-136`：
```c
static bool checkXhclMemChannel(struct ncclComm* comm, int connIndex, int channelId) {
  int collXhclChannelNum = getCollXhclChannelNum(comm);
  int p2pXhclChannelNum = getP2pXhclChannelNum(comm);
  if (connIndex == 0) { return channelId < collXhclChannelNum; }
  else if (connIndex == 1) { return channelId < p2pXhclChannelNum; }
}
```
- connIndex 0（collective 路径）用 COLL 阈值；connIndex 1（P2P 路径）用 P2P 阈值。

### 2.5 决策切 PCIe link memory（996-1010）

- `src/transport/p2p.cc:996-1010`：
```c
if (checkXhclMemChannel(connection->comm, connection->connIndex, connection->channelId) || connection->reqMemType == P2P_HIP_NOR_MEM) {
  NCCLCHECK(ncclP2pAllocateShareableBuffer(...));  // 普通 shareable buffer 路径
  p2pBuff->memType = P2P_HIP_NOR_MEM;
} else {
  p2pBuff->directPtr = NULL;
  NCCLCHECK(rocmAllocPcieLinkMem(connection->comm->hsaAgent, size, &p2pBuff->directPtr));  // PCIe link memory
  if (rocm_hsa_amd_ipc_memory_create(...) != ncclSuccess) { WARN(...); return ncclInternalError; }
  p2pBuff->memType = P2P_HSA_PCIE_MEM;
  INFO("rank %d connIndex %d channelId %d collXhclNum %d p2pXhclNum %d enable pcie mem channel", ...);
}
```
- 逻辑：在 connIndex 未超阈值 / 请求本来就是普通路径 → 用普通 shareable buffer；否则 → 分配 PCIe link memory（`rocmAllocPcieLinkMem`）。

### 2.6 其他 P2P 类型/内存处理相关行号

- `src/transport/p2p.cc:538/540/561`：p2pBuff memType 处理分支（NOR vs PCIE）
- `src/transport/p2p.cc:738`：recv proxy reqMemType = NOR
- `src/transport/p2p.cc:891-893`：resources recvMemType 分支
- `src/transport/p2p.cc:974`：p2pBuff->memType = NOR（默认）
- `src/transport/p2p.cc:1003`：`rocmAllocPcieLinkMem` 分配 PCIe link memory
- `src/transport/p2p.cc:1069/1071/1090-1092`：resMemType 分支处理

## 3. 关键文件源码定位速查

| 文件 | 用途 | 关键行号（真实） |
|---|---|---|
| `src/include/param.h` | NCCL_PARAM/RCCL_PARAM 宏定义 | `20-29`（NCCL_PARAM）、`31-44`（RCCL_PARAM） |
| `src/include/rccl_vars.h` | RCCL 环境变量声明区 | `28`（EnableHipGraph 唯一声明） |
| `src/init.cc` | RCCL 初始化 + 环境变量定义 | `128-136`（MSCCLPP 等）、`698-702`（跟踪）、`1009-1011`（P2P/LL128）、`1062`（注入）、`1638`（CollSimpleChannels）、`2478`（栈） |
| `src/enqueue.cc` | 协作任务调度/批处理 | `658`（IntraNetThreshold）、`945/955/961`（P2P 批处理） |
| `src/graph/connect.cc` | 通道连接/通道数约束 | `711`（OutputTrees）、`527-550`（min/max Nchannels）、`807-822`（MAXCHANNELS 约束） |
| `src/graph/paths.cc` | 路径/内网 | `828`（EnableIntranet） |
| `src/graph/rome_models.cc` | Rome 模型 | `1267`（ModelReversalDisable）、`1907`（DisableRailTrees） |
| `src/graph/search.cc` | 拓扑搜索 | `1043`（ModelMatchingDisable） |
| `src/transport/p2p.cc` | PCIe 混合链路核心 | `119-120`（XHCL channel 数）、`122-136`（阈值函数）、`996-1010`（切 PCIe link memory） |

## 4. 与既有 dcu-rccl-source 的关系

`dcu-rccl-source` SKILL.md 主文档基于**内部文档提取的机制解析**，讲解 PCIe 混合链路原理与三个环境变量的协同。本 reference 提供**26.04.2 真实源码树地图 + 环境变量真实定义位置 + 关键代码行号**，两者互补：

- 新环境变量（`P2P_BATCH_ENABLE`/`P2P_BATCH_THRESHOLD`/`P2P_NET_DISABLE`/`P2P_NET_THRESHOLD`/`INTRANET_THRESHOLD`）在 SKILL.md 主文档未覆盖，本 reference 补齐
- 源码纪律升级：主文档说"行号可能随版本漂移"，本 reference 给出**26.04.2 固定版本**的真实行号
- 实际改代码/查逻辑时，优先用本 reference 的行号核验（比主文档的漂移行号可靠）

## 5. 使用注意事项

- 本 reference 不存性能数字、不存逐行日志——只沉淀源码地图。
- 行号基于 26.04.2 分支；换版本前先 `grep` 确认目标版本的相同行号。
- 与 `dcu-rccl-test`（实测调优）衔接：调 PIe 混合链路 env 后仍用 `dcu-rccl-test` 实测留痕。
- 高危操作谨慎，优先测试机验证。