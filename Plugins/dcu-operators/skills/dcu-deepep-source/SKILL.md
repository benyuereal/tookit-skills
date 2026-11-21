---
name: dcu-deepep-source
description: DeepEP DCU 适配源码结构——MoE 专家并行 dispatch/combine（Buffer + low_latency 路由策略 + DTK 链接）+ 多节点带宽测试方法与实测参考线
version: 1.0.0
author: DCU Engineer
license: MIT
kind: knowledge
---

# DeepEP DCU 适配源码索引（dcu-deepep-source）

> 定位：**DeepEP 是 MoE 专家并行（Expert Parallelism）的 dispatch/combine 通信库**，海光移植到 DCU（HIP）。本 skill 讲 DeepEP 在 DCU 上的包结构、Python API、low-latency 路由策略、DTK 链接。
> 与 `dcu-aiter-moe` / `dcu-lightop-moe` 的关系：那两个讲 **MoE GEMM 算子**（专家内计算）；DeepEP 讲 **专家间通信**（dispatch/combine，跨卡/跨节点 token 路由）。MoE 全链路 = DeepEP（通信）+ AITER/LightOp/DeepGEMM（专家 GEMM）。
> 实测来源：`deep_ep-1.1.0+das185.dtk2604.torch2100.2608131245.g317bc7-cp310` wheel（release/dtk2604 源），解包 11 文件 + ldd 实测。

## §0 定位：DeepEP 是什么

> 上游锚点（2026-10-12）：`HYGON-AI/DeepEP-das` 锚点 commit `da6ca24`；同域新增 `HYGON-AI/UltraEP-das`（MoE dispatch/combine 上游 UltraEP 下游），与 DeepEP 同域，用到再展开。

DeepEP = MoE 的 **all-to-all 通信库**（dispatch：token 按 topk 路由到专家所在卡；combine：专家输出聚合回原卡）。两条路径：
- **normal（高吞吐）**：`dispatch`/`combine`，prefill 用，RDMA + NVLink/PCIe。
- **low-latency（低延迟）**：`low_latency_dispatch`/`low_latency_combine`，decode 用，分层优化 + 路由策略。

**DCU 适配**：海光把 DeepEP 的 CUDA kernel 移植到 HIP，`deep_ep_cpp.so` 链接 DTK 的 `libgalaxyhip.so.5` + `libhipblaslt.so.0` + `libhsa-runtime64` + `libmpi`（多节点）。

## §1 源码地图（wheel 实测目录）

```
deep_ep/
├── __init__.py              # 导出 Buffer / EventOverlap / low_latency_policy / Config
├── buffer.py                # ★ Buffer 类（53KB，dispatch/combine/low_latency 全 API）
├── low_latency_policy.py    # ★ low-latency 路由策略（23KB，RouteMode/DemandClass/RouteRule）
├── utils.py                 # EventOverlap（CUDA/HIP event 封装）
├── version.py
└── deep_ep_cpp.cpython-310-x86_64-linux-gnu.so   # ★ 编译产物（76MB，HIP kernel + DTK 链接）
```

## §2 Buffer 类 API（buffer.py，精确落点）

| 方法 | 行号 | 作用 |
|---|---|---|
| `class Buffer` | `buffer.py:19` | ★ 主类（持 RDMA buffer + 通信 stream） |
| `Buffer.__init__` | `buffer.py:38` | 初始化（group/num_sms/RDMA size） |
| `Buffer._setup_device_hca_mapping` | `buffer.py:159` | 设备-HCA 映射（多节点 RDMA） |
| `Buffer.is_ll_layered_opt_enabled` | `buffer.py:194` | low-latency 分层优化开关（group_size） |
| `Buffer.is_ll_layered_develop_enabled` | `buffer.py:205` | low-latency 分层开发开关 |
| `Buffer.ll_layered_opt_token_nums_threshold` | `buffer.py:212` | 分层优化 token 数阈值 |
| `Buffer.set_num_sms` | `buffer.py:221` | 设 SM 数 |
| `Buffer.capture` | `buffer.py:235` | CUDA graph capture（返回 EventOverlap） |
| `Buffer.get_low_latency_rdma_size_hint` | `buffer.py:245` | low-latency RDMA size 提示 |
| `Buffer.get_comm_stream` | `buffer.py:273` | 通信 stream |
| `Buffer.get_local_buffer_tensor` | `buffer.py:285` | 本地 buffer tensor |
| `Buffer.get_dispatch_config` | `buffer.py:319` | dispatch Config（按 num_ranks） |
| `Buffer.get_combine_config` | `buffer.py:348` | combine Config（按 num_ranks） |
| `Buffer.get_dispatch_layout` | `buffer.py:377` | dispatch layout |
| **`Buffer.dispatch`** | `buffer.py:422` | ★ normal dispatch（token 路由到专家卡） |
| **`Buffer.combine`** | `buffer.py:607` | ★ normal combine（专家输出聚合） |
| `Buffer.internode_dispatch` | `buffer.py:670` | 跨节点 dispatch |
| `Buffer.internode_combine` | `buffer.py:813` | 跨节点 combine |
| `Buffer.clean_low_latency_buffer` | `buffer.py:865` | 清理 low-latency buffer |
| **`Buffer.low_latency_dispatch`** | `buffer.py:883` | ★ low-latency dispatch（decode 用） |
| **`Buffer.low_latency_combine`** | `buffer.py:1009` | ★ low-latency combine（decode 用） |
| `Buffer.get_next_low_latency_combine_buffer` | `buffer.py:1091` | 取下一个 low-latency combine buffer |

## §3 low-latency 路由策略（low_latency_policy.py，精确落点）

low-latency 路径的核心是**路由决策**（token 走本地还是远端、走哪条 RDMA 路径）：

| 类/函数 | 行号 | 作用 |
|---|---|---|
| `class LowLatencyRouteMode(IntEnum)` | `low_latency_policy.py:20` | 路由模式枚举 |
| `class LowLatencyDemandClass(IntEnum)` | `low_latency_policy.py:31` | 需求类别枚举 |
| `class LowLatencyPeerLoad` | `low_latency_policy.py:40` | peer 负载（`from_counts`/`from_peer_record_counts`/`from_global_peer_record_counts`） |
| `class LowLatencyDemandHint` | `low_latency_policy.py:76` | 需求提示（`remote_record_ratio`/`active_remote_peers`/`zero_remote_peers`） |
| `class LowLatencyRouteRule` | `low_latency_policy.py:221` | 路由规则（`guaranteed_saved_remote_records`/`matches`） |
| `class LowLatencyRouteDecision` | `low_latency_policy.py:284` | 路由决策结果 |
| `normalize_low_latency_demand_hint` | `low_latency_policy.py:336` | 归一化需求提示 |
| **`choose_low_latency_route`** | `low_latency_policy.py:488` | ★ 选 low-latency 路由（num_nodes/hidden/tokens） |
| **`select_low_latency_route`** | `low_latency_policy.py:527` | ★ 应用 low-latency 路由选择 |

> 调 low-latency 性能：先看 `choose_low_latency_route`/`select_low_latency_route` 的路由决策逻辑，再调 `Buffer.is_ll_layered_opt_enabled`/`ll_layered_opt_token_nums_threshold` 分层参数。

## §4 DTK 链接（deep_ep_cpp.so，ldd 实测）

`deep_ep_cpp.cpython-310-x86_64-linux-gnu.so`（76MB）链接：
- **`libgalaxyhip.so.5`**（DTK HIP runtime，非 ROCm libamdhip64）
- `libhipblaslt.so.0`（DTK hipBLASLt）
- `libhsa-runtime64.so.1`（HSA runtime）
- `librocm-core.so.1` / `librocm_smi64.so.2`
- `libmpi.so.40`（多节点 MPI）
- `libc10.so` / `libtorch.so` / `libtorch_hip.so` / `libc10_hip.so`（torch HIP）

> 同 triton/ORT 的 DTK 库替换规律：`libgalaxyhip.so.5` 是 DTK 主 HIP 库。

## §5 与 vLLM/SGLang 的衔接

- **vLLM**：vllm-hcu 的 `vllm__model_executor__layers__fused_moe__prepare_finalize__deepep_{ht,ll}.patch.py` 是 DeepEP 接入 vLLM MoE 的 patch 落点（HT=high-throughput prefill，LL=low-latency decode）。见 `dcu-vllm-hcu-source` §7。
- **SGLang**：SGLang 的 MoE-EP 路径用 DeepEP（见 `dcu-sglang-source` 的 MoE-EP 证据）。
- **MoE 全链路**：DeepEP（dispatch/combine 通信）→ AITER/LightOp/DeepGEMM（专家 GEMM）。改 MoE 性能先分清是通信瓶颈（DeepEP）还是计算瓶颈（GEMM 库）。

## §6 与兄弟 skill 关系

- `dcu-aiter-moe` / `dcu-lightop-moe` / `dcu-deepgemm-source`：MoE GEMM 算子（专家内计算）→ DeepEP 是专家间通信，互补。
- `dcu-vllm-source` / `dcu-vllm-hcu-source`：vLLM MoE 层 → DeepEP 经 vllm-hcu patch 接入。
- `dcu-rccl-source`：RCCL 通信源码 → DeepEP 是 MoE 专用 all-to-all（比 RCCL 通用集合通信更特化）。
- `dcu-inference-infra`：推理 infra 总入口。

## §7 验证入口

```bash
# 安装（torch 标记对齐，本版 torch2100）
pip install deep-ep -i https://pypi.sourcefind.cn/release/dtk2604/

# 确认 DTK 链接（DCU 容器内）
ldd $(python3 -c "import deep_ep,os;print(os.path.dirname(deep_ep.__file__))")/deep_ep_cpp*.so | grep -E 'galaxyhip|hipblaslt|mpi'

# 基本用法（单卡 smoke）
python3 -c "
import torch, deep_ep
buffer = deep_ep.Buffer(group, num_qps_per_rank=1)
print('Buffer OK')
"
```

## §8 多节点带宽测试方法与实测参考线

> 本节来自现场实测（内部文档《【集群网络】DeepEP 带宽测试》），面向**跨节点 dispatch/combine 带宽/延迟验证**（对应 §0 的 low-latency 路径）。与 §1~§7 的源码结构互补：这里是"怎么跑测 + 测完拿什么当参考线"。

### 8.1 环境准备与依赖

- **镜像依赖**：Docker 镜像内需已正确安装并包含 `deepep`（wheel，见下）。
- **RoCE 网络环境变量**（若走 RoCE，必须在终端或启动脚本添加）：
  ```bash
  export ROCSHMEM_IB_GID_INDEX=0
  ```
- **拓扑配置文件 `topo.config`**：在运行目录下，按 **DCU Bus ID 与 Mellanox（mlx）网卡**的对应关系创建。格式为 `# DCU_busid  mlx网卡  序号`：
  ```
  <device-BDF>        mlx5_2    2
  <device-BDF>        mlx5_3    3
  <device-BDF>        mlx5_4    4
  <device-BDF>        mlx5_5    5
  <device-BDF>        mlx5_6    6
  <device-BDF>        mlx5_7    7
  <device-BDF>        mlx5_8    8
  <device-BDF>        mlx5_9    9
  ```

### 8.2 运行与测试命令（多机 low-latency）

运行前先检查并改 `run_test.sh` 中的主节点地址 `--master-addr`。底层 torchrun 形式：

```bash
torchrun --nproc-per-node=1 --nnodes=$NNODES --node-rank=$NODE_RANK \
  --master-addr="<主节点管理网 IP>" --master-port=1234 ./test_low_latency.py
```

多机示例（low-latency 模式，`run_test.sh <node_rank> lowl`）：
- 机器 A（Rank 0）：`bash run_test.sh 0 lowl`
- 机器 B（Rank 1）：`bash run_test.sh 1 lowl`

> 配套 `tests.zip` 内含测试脚本（`test_low_latency.py` / `run_test.sh`）。每节点 `--nproc-per-node=1` 起 1 进程测跨节点路径；`--master-addr` 填主节点管理网 IP，端口统一。

### 8.3 现场实测参考线（含基线环境）

> ⚠️ 以下为**某现场实测数据**，仅作"测完对比是否达标"的参考线，**不是通用于所有机型的基线**。对比前必须核对下表**基线环境**是否一致（软件版本、网络拓扑），否则数字不可比。
> 按纪律 skill 不固化性能数据，此处作为用户指定的"带基线的实测参考线"保留。

| 基线项 | 值 | 说明 |
|---|---|---|
| 软件包 | `deep_ep-1.1.0+dtk2604.torch2100.2607271907.ga77617-cp310-cp310-linux_x86_64.whl` | DTK 2604 + torch 2.1.0（torch2100）+ py3.10 |
| DTK / torch | dtk2604 / torch2100 | 与 §7 的 torch 标记对齐纪律一致 |
| 网络 | RoCE + `ROCSHMEM_IB_GID_INDEX=0` | 跨节点 RDMA |
| 拓扑 | 每节点 8 张 mlx5 网卡（mlx5_2~mlx5_9）对应 8 卡 | 见 topo.config |
| 节点 | 多机（master-addr <主节点管理网 IP>），`--nproc-per-node=1` | 单进程/节点测跨节点路径 |
| 机型 | **BW1100**（OAM，gfx938） | 用户补充，与《RCCL 通信测试》上篇一致 |

**实测数值（Dispatch + Combine 端到端）：**
- 正确性：DeepEP 全部正确性测试通过（底层数据路由与通信计算逻辑正常）。
- 端到端综合带宽：**35.79 ~ 35.85 GB/s**。
- 总体耗时：**614 ~ 616 µs**。
- 时延抖动：单卡最小耗时 548~569 µs，最大 652~683 µs，极差控制在 **130 µs 以内**（说明跨节点 RDMA 链路稳定）。

> Dispatch 与 Combine 分阶段拆解的耗时/带宽占比见原报告截图（未内联到本 skill）。

### 8.4 踩坑（关键）

1. **`topo.config` 中的卡网映射错误会导致 DeepEP 及分布式通信测试性能极低**。当卡和网卡拓扑错位时，底层驱动可能无法直接建立 Peer-to-Peer (P2P) DMA 通道，数据被迫退化为**通过系统内存（Host Memory）进行二次拷贝**，导致耗时翻倍（Dispatch/Combine 耗时显著增加）。
   > 生成 topo.config 前务必先用 `lspci`/`ibstat` 核对 DCU Bus ID 与 `mlx5_x` 网卡的**真实对应关系**，不要凭顺序猜测。同 `dcu-roce-network` 的"IP 与 MAC 一一对应核对清楚"是同一类坑。

## 使用须知

- DeepEP 是**通信库**（dispatch/combine），不是 GEMM 库——MoE 性能问题先分清通信 vs 计算。
- **normal vs low-latency**：prefill 用 `dispatch`/`combine`（高吞吐），decode 用 `low_latency_dispatch`/`low_latency_combine`（低延迟 + 路由策略）。
- **DTK 链接**：`libgalaxyhip.so.5` + `libhipblaslt` + `libmpi`（多节点），DTK 安装不完整会加载失败。
- **torch 标记对齐**：本版绑 torch2100，装到 torch 2.9 会 ABI 不匹配（见 `dcu-das-hub-api` §5）。
- low-latency 路由策略（`choose_low_latency_route`）是 decode MoE 性能关键，调优先看路由决策。
- 默认**不存性能数据/基线**（数字用 `dcu-op-optimization` 模板留痕）；唯一例外是 §8.3 的"带基线环境的现场实测参考线"（用户指定保留，对比前须核对基线一致）。
- 高危操作谨慎，优先测试机验证。
