---
name: dcu-rccl-test
description: RCCL 多卡互联带宽测试：单/多节点 AllReduce 性能测试与排障
version: 1.1.0
author: DCU Engineer
license: MIT
kind: tool
---

# DCU RCCL 集合通信测试

## 0. 操作范围门（scope gate，执行前必读）

本 skill 涉及的一切执行动作（占卡跑测试、SSH 到节点、起容器编译/运行 rccl-tests）**默认只允许作用于"本会话自己启动"的资源**：

- **卡**：`hy-smi` 先看占用，只选空闲卡跑测试；卡被占时有界轮询或先问用户，**绝不杀他人的测试/训练进程腾卡**。
- **节点**：SSH 只登录用户已授权的目标机；多节点测试前确认各节点卡空闲；不改他人的网络配置（sysctl / NIC / 交换机口）。
- **端口/文件**：测试临时文件放自己的工作目录，用完清理；不动他人 home 下的东西。
- 例外（须动他人资源才能完成任务）：逐次向用户报确切命令获批准后执行；长期成立的例外登记到下方白名单，其余情形仍按默认规则。

**例外白名单（经用户批准后逐条登记）**：暂无

## Overview

> 📌 **遇到具体报错先查排查经验库**：NCCL/RCCL 通信卡死、带宽低、`NCCL_SIMPLE_CHANNELS`/`NCCL_RINGS` 配置、dcu_megatron 默认 NCCL 不适配等具体现象的处理经验（含 env/修复命令），统一收在 **`dcu-ontrack-experience`** 这个 skill 里。本 skill 只讲测试流程与编译参数，具体踩坑手法请去那里按"现象关键词"反查。

RCCL（ROCm Collective Communication Library）是 DCU 的集合通信库，实现 all-reduce / all-gather / reduce-scatter / all-to-all 等通信模式，可在 PCIe、xGMI、InfiniBand/TCP 上实现高带宽通信。本 skill 用于**量化测评 DCU 多卡/多机互联带宽**，是交付验收、标前验证、性能对比的必做项。

工程地址：https://github.com/ROCm/rccl-tests

> 来源：内部测试手册《rccl-test 通信测试-260610》（王凯雄，修订至 2026-06-10），本 skill 为其结构化整理版。

## 速查表：产品型号 → 部署关键参数 / DTK 版本 → 部署差异（已下沉）

选型速查表两张 见 [references/quick-ref-models-dtk.md](references/quick-ref-models-dtk.md)。

## When to Use

- 需要测评 DCU 单节点多卡 / 多节点多卡的互联带宽（all-reduce、alltoall）
- 交付验收、标前验证、售后排查中需要量化集合通信性能
- 双机/多机训练前的互联基线确认
- 排查 MPI_INIT 失败、NCCL 报错等集合通信问题

Don't use for：纯推理性能测评（见 dcu-docker 测评章节）；容器内 SSH 免密配置本身（见 dcu-docker）。

## ⚠️ 基本环境准备（来自内部测试手册，测前必查）

1. **BIOS 关闭 ACS 和 IOMMU（仅 PCIe 卡适用）** —— 否则 PCIe P2P 受限、通信带宽不达标。**此要求主要针对 PCIe 形态卡**（K100-AI、BW10、BW100、BW150 等）；**OAM 产品（BW1000、BW1100）基本不受影响，无需为此改 BIOS/内核**。
   用 dcu-env 的 `../dcu-env/scripts/check_iommu_acs.sh` 检测；未全关则提示用户先关再测（见 dcu-env）。
2. **多机测试需设免密通信** —— 双机/多机 mpirun 依赖节点间 root 免密（物理机或容器免密，见 dcu-ssh / dcu-docker）。
3. **重装驱动后需重启设备** —— 驱动升级（尤其带 vbios 更新）必须重启物理机，否则驱动未完全生效。

> 以上三条是 rccl-test 的前置硬条件，违反会直接导致带宽不达标或测试失败。

## ⚠️ 强制前置步骤：检查磁盘剩余空间（拉镜像 / clone / 编译会占额外空间）

RCCl 测试链路里**多个环节会持续吃盘**，且物理机上常已有数十个业务容器（实测某机型同时存在 19 个容器），叠加新测试容器 + 镜像 + 编译产物极易把盘写满，导致 `docker pull` 失败、编译中断、日志写不进去。

**任何 RCCL 测试开始前，必须先确认宿主机/目标盘剩余空间足够。** 这一步在"指定镜像"之后、起容器之前做。

### 检查命令（物理机/宿主机执行）
```bash
df -h / /data /public          # 看目标盘剩余（按实际挂卷所在盘替换）
docker system df               # 看 docker 镜像/容器/卷已占用多少
# 找出占空间最多的容器/镜像（定位可清理项）：
docker ps -as --format 'table {{.Names}}\t{{.Size}}' | sort -k2 -h
docker images --format 'table {{.Repository}}\t{{.Tag}}\t{{.Size}}' | sort -k3 -h
```

### 空间占用预估（心里有数）
| 环节 | 大概占用 | 备注 |
|---|---|---|
| 拉取一个 DTK/推理测试镜像 | 10 ~ 30 GB | 镜像常带完整 DTK + 框架 |
| 起测试容器（不挂单独卷，写 layer） | 数 GB（可增长） | 容器内 apt/编译会落在本容器可写层 |
| `git clone` + 编译 rccl-tests（`build/`） | 2 ~ 5 GB | 在容器内 `/root/rccl-tests` 或挂载目录 |
| `topo_tools` 等其他 clone | < 1 GB | 用完即清 |
| 测试日志（重定向到 `/tmp/x.log`） | 通常 < 100 MB | 但若爆盘会写失败 |

### 阈值与处置
- **剩余 < 20 GB**：黄色警告，先确认无更大文件要写；建议先清理再继续。
- **剩余 < 10 GB**：**拒绝继续**，先释放空间：
  ```bash
  docker image prune -f                       # 清 dangling 镜像
  docker rm -f <不再用的旧测试容器>           # 删掉已结束的临时容器（注意别删业务容器）
  # 或清理 rccl-tests 旧 build、/tmp 下大日志等
  ```
- 目标盘以实际 `docker info | grep -i 'docker root'` 所在文件系统为准（常是 `/var/lib/docker` 所在盘，不一定是 `/`）。

### 用完清理（避免长期累积）
- 测试结束、日志 `docker cp` 到宿主机并取回后，删掉测试容器：`docker rm -f <CONTNAME>`。
- 若镜像为临时专用镜像且不再复用，可 `docker rmi <IMG>`（确认不影响他人）。
- 容器内 `/root/rccl-tests`、`/root/topo_tools` 等 clone 目录，容器删了即随之消失；若挂在宿主机卷上则需手动 `rm -rf`。

## ⚠️ 强制前置步骤：测试前必须先由用户指定 Docker 镜像

**任何 RCCL 测试（单节点/双节点）开始前，必须先让用户明确指定用于测试的 Docker 镜像名**。
后续所有编译、运行都**固定基于该镜像启动的容器**，不临时复用无关容器、不臆测 DTK 版本。

理由：
- DCU 测试强依赖镜像内的 DTK 版本 / rccl / MPI / gfx 架构，不同镜像结果不可比；
- skill 速查表未覆盖 BW100/BW150 等型号的 gfx 与编译 commit，须以用户指定镜像内的实际环境为准；
- 避免误用他人正在运行的容器（实测某机型上同时存在 19 个业务容器），污染现场或抢占资源。

执行流程：
1. **向用户询问并确认测试镜像**（如 `hygon/dtk:25.04.1-*u1-ubuntu22.04`、`image:25.04.1-...` 等完整镜像名）；
   - 若用户未提供，先 `docker images` 列出可用镜像让用户选，禁止自行假设；
2. 基于该镜像 `docker run` 一个**干净的专用测试容器**（不要复用现有业务容器），挂 `--network=host`、`-v /opt/dtk`、`-v /opt/mpi`、GPU 设备透传；
3. 在专用容器内 source 镜像自带 DTK 环境，探明 gfx / rccl / MPI 实际路径后再编译；
4. 全程记录所用镜像名 + DTK 版本，写进测试报告（以便复现与对比基线）。

> 例：用户指定 `image:25.04.1-ubuntu22.04-dtk-v5.2-20260409`，则后续所有命令均在该镜像起的容器内执行。

## ⚠️ 测试结果输出规范（强制）

向用户汇报 rccl-test 结果时，**必须遵守**：

1. **输出全部原始日志**：测试指令（`all_reduce_perf` / `alltoall_perf` 等）的完整 stdout 一字不漏贴出，包括表头、逐 size 行、`#wrong`、`Out of bounds` 等，**不得裁剪、不得只摘峰值行**。
2. **不计算 / 不强调均值**：禁止用 `Avg bus bandwidth`、算平均 busbw 等方式做汇总结论；只呈现指令原样输出。
3. **不自行解释异常、不分析**：若日志出现 `Launch params larger than launch bounds`、redop=none 等告警，原样保留，**不在汇报里做分析、不判定"脏数据"、不替用户下结论**。用户明确要求"测试完成提供结果即可，不需要分析"——只给原始日志，不附加解读。
4. 汇报结构建议：直接给**完整日志块**即可；如确有需提示项，单独列在日志之后并明确标注"观察/待确认"（与结果分离，且非强制）。

> 理由：均值会掩盖逐 size 行为（小包走延迟、大包走带宽），且初测/重测环境差异可能导致均值误导；完整日志才能让用户自行判断。

## 测试指标介绍

| 字段 | 含义 |
|---|---|
| size | 通讯字节总数量 |
| count | 发包总数量 |
| type | 数据类型 |
| redop | 操作类型 |
| time | 总耗时 |
| algbw | 算法带宽 = size / time |
| busbw | 总线带宽（消除 rank 数影响，可与硬件峰值比较） |

**busbw 换算公式（n = GPU 数量）：**
- AllReduce：`algbw * 2*(n-1)/n`
- ReduceScatter / AllGather：`algbw * (n-1)/n`
- Broadcast / Reduce：`algbw * 1`

> 20260408 手册给出完整**总线带宽校正系数表**（与上式等价，列于此便于查表）：

| 集合通信操作 | 总线带宽校正系数 |
|---|---|
| AllReduce | 2*(n-1)/n |
| ReduceScatter | (n-1)/n |
| AllGather | (n-1)/n |
| Broadcast | 1 |
| Reduce | 1 |
| AlltoAll | (n-1)/n |

> 参考：https://github.com/ROCm/rocm-systems/blob/develop/projects/rccl-tests/doc/PERFORMANCE.md （algbw 反映应用感知吞吐；busbw 反映底层物理链路真实速率，用于对比硬件峰值定位瓶颈。）

> In-place / out-of-place 通常结果一致，后者仅为功能测试添加。

## 集合通信性能估算方法（判瓶颈用）（已下沉）

延迟/吞吐分界、Ring vs Tree 选型、分层 max、Bisection、SHARP、与推理指标关系 见 [references/bandwidth-estimation.md](references/bandwidth-estimation.md)。

## GPU_TARGETS 说明（按 DCU 型号）

| DCU 型号 | GPU_TARGETS |
|---|---|
| Z100L | gfx906 |
| K100 | gfx926 |
| K100-AI | gfx928 |
| BW1000 | gfx936 |
| BW1100 | gfx938 |

> 编译和设置环境变量时必须对应正确 gfx 架构，否则编译失败或运行异常。

## 代码拉取与编译

构建产物在 `rccl-tests/build/` 目录。

### BW1000 / K100-AI（推荐方法）
```bash
git clone https://github.com/ROCm/rccl-tests.git -b master
cd rccl-tests/
# 回退到稳定 commit（老分支不支持 fp8，但 DTK-25.04/25.04.1 可用）
git reset --hard 5e838ad9df47079e0e586ed38049f4f579ea462d

# （可选）解决 block 警告：src/Makefile 第32行后加 --gpu-max-threads-per-block=1024

# 编译（ROCM_HOME / MPI_HOME 按实际 DTK / MPI 路径替换）
make ROCM_HOME=/opt/dtk NCCL_HOME=/opt/dtk/rccl \
     CUSTOM_RCCL_LIB=/opt/dtk/rccl/lib/librccl.so \
     MPI=1 MPI_HOME=/opt/mpi -j8
```
> 构建方法2（要求 DTK-25.04.1）：`GPU_TARGETS=gfx936 HIPCC=hipcc MPI=1 MPI_HOME=/opt/mpi make -j -C src build`

### BW1100
```bash
git clone https://github.com/ROCm/rccl-tests.git -b master
cd rccl-tests
git reset --hard 66e513c24ff42394f5a0c1781f5868da7e094dd1

# （可选）src/Makefile 第32行后加 --gpu-max-threads-per-block=1024

./install.sh --mpi --mpi_home /opt/mpi \
  --rocm_home /opt/dtk \
  --rccl_home /opt/dtk/rccl \
  --hip_compiler hipcc \
  --gpu_targets gfx938
```

## 测试容器启动（强制，单/双节点通用）

RCCL 测试**必须在容器内编译并运行**（物理机直接编译会因 `/opt/dtk` 软链版本错位、
C++ 标准库路径缺失而失败；且业务机通常已有他人容器，需隔离）。

### 启动容器（关键：必须挂 /opt/hyhal）

```bash
docker run -d --name rccl-test-xxx --network=host --ipc=host \
  -v /opt/dtk:/opt/dtk -v /opt/mpi:/opt/mpi -v /opt/hyhal:/opt/hyhal:ro \
  --device=/dev/kfd --device=/dev/dri \
  --privileged <用户指定镜像> sleep infinity
```

> ⚠️ **`-v /opt/hyhal:/opt/hyhal:ro` 是必挂项**。漏挂会导致容器内 `/opt/hyhal/lib` 缺失，
> `hipGetDeviceCount` 返回 0（DCU 设备不可见），测试直接失败且无明确报错。
> 实测（2026-07-22）：漏挂时 DEVCOUNT=0；挂上后 DEVCOUNT=8 正常。

### gfx 架构查询（物理机执行，非容器）

```bash
# 物理机（宿主机）执行，需带 ROCM_PATH / LD_LIBRARY_PATH，否则 rocminfo 无输出
export ROCM_PATH=/opt/dtk-25.04      # 用物理机实际存在的 DTK 版本
export LD_LIBRARY_PATH=/opt/dtk-25.04/lib:$LD_LIBRARY_PATH
/opt/dtk-25.04/bin/rocminfo 2>&1 | grep -oiE 'gfx[0-9]{3}' | sort -u
```

> ⚠️ rocminfo **必须在物理机执行**，容器内 rocminfo 会把 DCU 误识别为 CPU Agent（拿不到 gfx）。
> 物理机若无输出，先 `ls /opt/dtk-*/bin/rocminfo` 确认版本路径，并加上述两个环境变量。

### 编译与运行位置

- **编译**：在容器内，`source /opt/dtk/env.sh` 后按型号用 make 法（见下）。
- **运行**：在容器内，8 卡用 `-g 8`（单节点）或 mpirun `-g 1`（双节点每进程1卡）。

### 一键流程脚本（推荐复用，避免重复踩坑）

`scripts/rccl_test_single_node.sh` 已固化上述全流程：物理机查 gfx → 起容器（挂 /opt/hyhal）→ 验 8 卡可见 → 查/清 `NCCL_NCHANNELS_PER_PEER` → 编译 → 跑 2/4/8 卡 all_reduce + 8 卡 alltoall → docker cp 日志。用法见脚本头注释。脚本**不含任何性能数值**，仅流程。

## 单节点多卡测试

### 建议环境变量
```bash
export HSA_FORCE_FINE_GRAIN_PCIE=1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   # 全卡可见；指定子集用 0,1 / 0,1,2,3 等，编号从 0 开始
export NCCL_TOPO_FILE=null
```

> ⚠️ **指定部分卡必须用 `HIP_VISIBLE_DEVICES` 限制卡号**（如 4 卡 = `0,1,2,3`，2 卡 = `0,1`），编号从 0 起。
> 不要只靠 `-g N` 让 RCCL 自行挑卡——显式设 HIP_VISIBLE_DEVICES 才能固定到指定物理卡，避免 RCCL 选到非预期卡序导致结果不可复现。

### ⚠️ 测试前必做：检查并清除 NCCL_NCHANNELS_PER_PEER

**进入测试容器后、跑任何 perf 指令前，先检查环境里是否已设置 `NCCL_NCHANNELS_PER_PEER`**（镜像/容器可能自带，如 `=2`）。若有，先 `unset` 再测试，避免非默认通道数污染基线对比：

```bash
echo "NCCL_NCHANNELS_PER_PEER=${NCCL_NCHANNELS_PER_PEER:-<unset>}"
# 若上面输出不是 <unset>（即已被设值），则：
unset NCCL_NCHANNELS_PER_PEER
```

> 理由：NCCL 通道数直接影响通信性能，镜像预置的 `NCCL_NCHANNELS_PER_PEER`（常见于某些 vllm/推理镜像）会让测试结果不可与"默认通道"基线对比。测试前统一清掉，保证每次测试环境一致。
> 注意：仅 alltoall 单机双卡场景才**按需显式设** `=16`（见下文），其余测试一律先 unset。

### ⚠️ 单节点覆盖清单（用户通常要求低卡数也测）
除 8 卡外，**用户常要求补测 2 卡 / 4 卡 all_reduce**（验证小拓扑带宽）。务必覆盖：
```bash
# （单机低卡覆盖：清 NCHANNELS_PER_PEER 后再测）
./build/all_reduce_perf -b 2 -e 2G -f 2 -g 2    # 2 卡（HIP_VISIBLE_DEVICES=0,1）
./build/all_reduce_perf -b 2 -e 2G -f 2 -g 4    # 4 卡（HIP_VISIBLE_DEVICES=0,1,2,3）
./build/all_reduce_perf -b 2 -e 2G -f 2 -g 8    # 8 卡
./build/alltoall_perf -b 2 -e 2G -f 2 -g 8      # 8 卡 alltoall
```
> 2 卡 / 4 卡 / 8 卡 all_reduce 均应覆盖；指定卡数时务必用 `HIP_VISIBLE_DEVICES` 限制到对应物理卡（见上）。

### all_reduce_perf（8 卡示例）
```bash
# （8 卡示例：已清 NCHANNELS_PER_PEER）
./build/all_reduce_perf -b 2 -e 2G -f 2 -g 8
```

### alltoall_perf（8 卡示例）
```bash
# （8 卡 alltoall：已清 NCHANNELS_PER_PEER）
./build/alltoall_perf -b 2 -e 2G -f 2 -g 8
```

> 单机双卡 alltoall 建议设 `NCCL_NCHANNELS_PER_PEER=16`。

## 双节点 16 卡测试

### 测试形式（任选一种）
- 基于双机**物理机**环境测试
- 基于双机**容器**环境测试（容器免密配置见 dcu-docker；本文 5.2/5.3 也有原始步骤）

### 前置条件
- 完成双机免密（物理机免密 或 容器免密，见 dcu-docker）
- 安装 MPI 工具，建议版本 >= 4.1.1

> ⚠️ **国产 IB 网卡（曙光 SHCA 等）注意**：上表 `NCCL_IB_HCA=mlx5_x` 是 Mellanox 写法。SHCA 等国产卡必须用厂商官方 RCCL 网络插件（`NCCL_NET_PLUGIN=.../librccl-net-shca.so`），且 `NCCL_IB_HCA` 不带 `:端口` 后缀、`NCCL_IB_PCI_RELAXED_ORDERING=0`、通道数两端显式锁死、topo 按节点独立——具体现象→修复条目见 `dcu-ontrack-experience` C 节，完整调优案例（含启动脚本）见 `dcu-perf-cases` 案例 14。

### 测试前准备（DTK-25.04.2 必做）
1. 查 HCA 卡数量：
   ```bash
   ibstat | grep mlx5_ | wc -l
   ```
2. 按实际数量改 topo 文件（双机都要改）：
   ```bash
   vim /opt/dtk/rccl/lib/topo_mapping_default.xml
   # 把 gfx936_8_x86_64_HygonGenuine_mlx5_10 中的 10 改成实际查询到的数量（如 11）
   ```

### 双机 16 卡 all_reduce
> 注意：物理机测试无需 `--mca plm_rsh_args "-p 2345"`（2345 是容器免密端口，按实际免密配置替换）；`NCCL_IB_HCA` 按实际网卡填。
```bash
# 已按上文"测试前必做"清除 NCCL_NCHANNELS_PER_PEER（两节点均需先检查 unset）

mpirun --prefix /opt/mpi \
  -np 16 -H node1:8,node2:8 --allow-run-as-root \
  --mca btl_tcp_if_include p14p2 \
  --mca plm_rsh_args "-p 2345" \
  -x NCCL_SOCKET_IFNAME=p14p2 \
  -x HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 -x ROCM_PATH -x LD_LIBRARY_PATH \
  -x NCCL_IB_HCA=mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_7,mlx5_8,mlx5_9,mlx5_10  \
  -x NCCL_NET_GDR_LEVEL=SYS -x NCCL_NET_GDR_READ=1 \
  ./build/all_reduce_perf -b 2 -e 2G -f 2 -g 1
```

### 双机 16 卡 alltoall
同 all_reduce，仅末行换成 `./build/alltoall_perf -b 2 -e 2G -f 2 -g 1`。

## 多机通信优化最佳实践（BW1000 / BW1100）

### 一、DCU 驱动优化（驱动版本 >= 6.3.30 建议开启 xdp_size）
```bash
cat /sys/module/hycu/parameters/xdp_size
hy-smi --setdriverparams xdp_size=15
# 停掉所有容器，确认卡无占用后再 loaddriver
hy-smi --loaddriver
cat /sys/module/hycu/parameters/xdp_size
```

### 二、topo mapping file 最优配置
使用内部工具（暂不支持 X7950H0 以外机型、mlx 以外网卡）：
```bash
git clone http://developer.sourcefind.cn/codes/tsoc/topo_tools.git
cd topo_tools
python generate_topo_mapping.py     # 自动生成 topo mapping 文件及用法
export NCCL_TOPO_MAPPING_FILE=/path/to/topo_mapping_custom.xml
```

### 三、推荐使用环境变量
```bash
export NCCL_TOPO_MAPPING_FILE=/models/rccl-tests-.../topo_tools/topo_mapping_custom.xml
# 用了 MAPPING_FILE 后双机理论不需设 NCCL_IB_HCA；性能仍低则禁用无用网卡：
# export NCCL_IB_HCA=^mlx5_0:1,^mlx5_5:1,^mlx5_6:1
export NCCL_NET_GDR_LEVEL=SYS
export NCCL_NET_GDR_READ=1
# ROCE 环境额外 2 项：
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=96    # cumulus 默认96；华三一般162（按交换机厂商值）
```

### 四、性能仍差的处理建议
提供 `HCU-bus.log`、`sys-tvvv.log`、DTK 版本给海光侧分析，获取优化后的 topo 文件；双机指定：
```bash
-x NCCL_TOPO_FILE=topo-input.xml     # 双机都需上传 topo 文件
```
获取日志：
```bash
hy-smi --showbus  > HCU-bus.log
lspci -tvvv > sys-tvvv.log
```

## 性能基准参考（各机型单机 + BW1000 双机 16 卡）（已下沉）

内部测试手册实测数据 见 [references/benchmark-results.md](references/benchmark-results.md)。

## 测试环境 / NCCL·RCCL 调优 env 速查（已下沉）

通用测试指令、单机调优 env、alltoall 单机双卡必做 unset、集群 mpirun 模板 见 [references/env-tuning-cheatsheet.md](references/env-tuning-cheatsheet.md)。

## 容器免密（双机测试前置）

双机容器测试前需配容器免密，详见 **dcu-docker**（标准方法为容器内 sshd 改非22端口 + 公钥互信 + mpirun 加 `--mca plm_rsh_args "-p <端口>"`）。

原始手册两种容器免密法（摘要）：
- **方法1**（依赖物理机已免密）：启动容器时 `--network=host` + `-v /root/.ssh:/root/.ssh`，容器 `.bashrc` 末尾加 `/usr/sbin/sshd -p 1234`，mpirun 加 `--mca plm_rsh_args "-p 1234"`。
- **方法2**（物理机无需免密）：容器内装 openssh-server，改 `sshd_config`（设端口 / 允许 root 登录 / 允许授权免密），`.bashrc` 加 `service ssh restart`，各容器生成密钥 + `/etc/hosts` 配节点名 + `ssh-copy-id -p 1234` 互信。

> 标准做法以 dcu-docker 为准；此处保留手册原步骤供对照。

物理机免密（附录5.1 摘要）：关 firewalld/iptables → 各机 `ssh-keygen -t rsa` → `/etc/hosts` 配管理网 IP 与主机名 → `ssh-copy-id` 互信（注意用管理网 IP，非 IB/ROCE 网）。

## Common Pitfalls

1. **GPU_TARGETS 选错** — BW1000/BW150=gfx936、BW1100=gfx938，错配编译失败或运行异常。BW150 与 BW1000 同 gfx936（2026-07-22 物理机 rocminfo 实测）。
2. **commit 版本不对** — master 分支更新后可能拉不下来/编译不过，BW1000/BW150 用 `5e838ad...`、BW1100 用 `66e513c...` 固定 commit。
3. **双机 topo_mapping_default.xml 的 HCA 数与实际不符（DTK-25.04.2）** — 必须按 `ibstat | grep mlx5_ | wc -l` 改 mlx5_N 的数字，否则性能异常。
4. **MPI_INIT failed / 进程不可达** — 查 IB/ROCE 网络通畅；或临时绕管用 `--mca pml ob1 --mca coll_hcoll_enable 0 -x NCCL_IB_DISABLE=1`（仅跑通流程，结果无参考意义）。
5. **NCCL internal error (common.cu.cpp:1283)** — 查 `NCCL_SOCKET_IFNAME` 是否设为正确的本机管理网口名。
6. **容器里跑测试性能偏低** — 确认 mpirun 的 `--mca plm_rsh_args` 端口与容器 sshd 端口一致；且用 `docker exec -d bash -ilc '...'` 启动以保证 DTK 环境变量（rocblas/hipblaslt）加载（见 dcu-docker 测评章节）。
7. **xdp_size 未开** — 驱动 >= 6.3.30 建议开 `xdp_size=15` 并 `loaddriver`（需先停容器、卡无占用）。
8. **未用 NCCL_TOPO_MAPPING_FILE** — BW1000/1100 双机建议用 topo_tools 生成最优 mapping 文件，性能可明显提升。
9. **⚠️ 容器漏挂 `/opt/hyhal` → DCU 不可见** — 起测试容器必须 `-v /opt/hyhal:/opt/hyhal:ro`，否则容器内 `/opt/hyhal/lib` 缺失，`hipGetDeviceCount` 返回 0（DEVCOUNT=0），测试静默失败。实测（2026-07-22）：漏挂 0 卡、挂上 8 卡。
10. **src/Makefile 的 `--gpu-max-threads-per-block=1024` patch 实测未生效** — 按手册在 `src/Makefile` 加该 flag 后，运行时仍报 `Launch params (512,1,1) larger than launch bounds (256)` 警告（dtk26.04, 2026-07-22）。说明 sed 替换未命中实际编译行或该 flag 被覆盖；后续如需消除警告，应先 `grep` 确认 HIPCCFLAGS 真正含该参数，或改用 `GPU_TARGETS=... HIPCC=hipcc make ...` 显式传参。
11. **rocminfo 查 gfx 必须物理机 + 带环境变量** — 容器内 rocminfo 会把 DCU 误识别为 CPU Agent（拿不到 gfx）；物理机执行还需先 `export ROCM_PATH=/opt/dtk-<ver>` `LD_LIBRARY_PATH=...` 否则无输出。详见"测试容器启动"章节。
12. **测试日志严禁写入 skill** — 完整测试日志（all_reduce/alltoall 逐行输出）只汇报给用户，不固化进 skill；skill 仅沉淀经验/踩坑/规范，不存性能数据。
13. **⚠️ 从容器取日志：容器/宿主机 /tmp 不共享** — 测试输出重定向到容器内 `/tmp/x.log` 后，必须 `docker cp 容器名:/tmp/x.log /tmp/x.log` 拷到宿主机，再 scp 到本地；直接 `scp 容器IP:/tmp/x.log` 会因容器 /tmp 与宿主机 /tmp 隔离而报 "No such file or directory"。实测（2026-07-22）踩过两次。
14. **⚠️ sshpass scp 子命令需重新 export SSHPASS** — 在一条复合 ssh 命令里先 `export SSHPASS=...` 再 `ssh ...; scp ...`，scp 作为新子 shell 不会继承当前 shell 的 SSHPASS 环境变量，会报 "SSHPASS: -e option given but SSHPASS environment variable not set"。解决：每条 scp 前单独 `export SSHPASS=$(...)`，或把密码写进临时文件用 `-f`。

## Verification Checklist

- [ ] 目标盘剩余空间已查（df -h + docker system df），< 10G 已清理再继续、< 20G 已确认
- [ ] 已按 DCU 型号选对 GPU_TARGETS 并 checkout 正确 commit 编译
- [ ] build/ 下生成 all_reduce_perf / alltoall_perf
- [ ] 单节点：环境变量已设（HSA_FORCE_FINE_GRAIN_PCIE 等），8 卡测试跑通
- [ ] 双节点：免密完成（物理机或容器），MPI >= 4.1.1
- [ ] 双机（DTK-25.04.2）：topo_mapping_default.xml HCA 数已按实际改正
- [ ] mpirun 的 plm_rsh_args 端口 / NCCL_IB_HCA / NCCL_SOCKET_IFNAME 已按实际网卡配置
- [ ] 性能与参考基线对比（all_reduce busbw 应接近 117~128 GB/s）
- [ ] （BW1000/1100）已开 xdp_size + 用 topo_tools 生成 mapping 文件优化

## 20260408 版手册互补内容（已下沉）

拓扑导出 / 四机 32 卡 / 容器免密完整 docker run / launch bounds 报错修法 见 [references/manual-20260408-supplement.md](references/manual-20260408-supplement.md)。

## References

- 原始手册：《rccl-test 通信测试-260610》（王凯雄）
- 官方测试手册（互补来源）：《RCCL通信测试手册-20260408》（内部文档 file_id `<internal-file-id>`；含指标说明截图/编译结果/ssh_config/报错截图，已按纪律剔除 base64 内联图片，仅取文字命令表格；提供 NCCL_TOPO_DUMP_FILE 导出、四机 32 卡完整 mpirun、容器免密完整 docker run、Launch params kernel 报错两种编译修法、NCCL_NET_GDR_LEVEL PHB/SYS 差异）
- 工程：https://github.com/ROCm/rccl-tests
- topo_tools：http://developer.sourcefind.cn/codes/tsoc/topo_tools.git
- **ducom-test 官方通讯性能测试工具包**（z-file Cat5「工具包→通信性能测试」，配套 DTK 25.04.3 手册）：开箱即用的通讯测试打包，与本 skill 的 rccl-tests 源码编译路线互补（要改源码/加集合通信用本 skill，快速出数用 ducom-test）。
- 官方文档中心《RCCL-TEST 指导手册》：版本树 `42c18d04-420d-11f1-bde9-0242ac150003`（单叶全文 `8d105702-420d-11f1-bde9-0242ac150003`），`GET https://developer.sourcefind.cn/api-doc/document/DetailDocument?UUID=<uuid>` 匿名取全文（2026-10-14 实测）
- 关联 skill：dcu-docker（容器免密 / 测评 exec 方式）、dcu-env（DTK 环境变量）、dcu-ssh（远程连接）、dcu-rccl-source（PCIe 混合链路优化原理，与本文 §单机调优 env 的实测值呼应）
