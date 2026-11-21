---
name: dcu-dtk-layout
description: "DCU DTK目录结构与自带工具脚本速查"
version: 1.3.0
author: DCU Engineer
license: MIT
---

# dcu-dtk-layout — DTK 目录结构与自带工具脚本速查

本 skill 是 **DTK（DCU Toolkit）内部“地图”**。它回答的是：
- `/opt/dtk` 里都有什么目录、各占多大？
- DTK 自带那些工具脚本（hipcc / hipconfig / rocminfo / gpurun / clang-ocl / hipgdb / extractkernel …）分别干嘛、源码怎么写？
- 编译/运行时要 source 哪些环境变量、用到哪些库？

> 与 `dcu-dtk` 的区别：`dcu-dtk` 讲「怎么下载/安装/激活 DTK」（仅物理机需要）；
> 本 skill 讲「DTK 装好后里面长什么样、每个工具怎么用」。两者互补，本 skill 偏实测速查。

## When to Use

- 想在容器内用 DTK 的某工具（hipcc / rocminfo / hipprof / gpurun / clang-ocl …）但不确定路径或参数
- 排查编译/运行问题：hipcc 找不到、LD_LIBRARY_PATH 缺库、架构(gfx)不匹配
- 想看懂 `gpurun` 怎么绑 NUMA 核 / 切 CU 掩码（多卡多 rank 性能优化）
- 想确认 DTK 版本、支持的 gfx 架构、cuda 兼容层怎么开
- 想把 CUDA 代码移植到 HIP（hipify 工具集在哪）

Don't use for：物理机首次安装/激活 DTK（用 `dcu-dtk`）；驱动问题（`dcu-driver`）；设备可见性（`dcu-hy-smi`）。

## 实测基线（重要）

- 来源设备：`<device-B> 设备（BW100 8 卡）`，容器 `<container_vllm015>`
- 镜像：`<internal_registry>/dcu/admin/base/custom:vllm015-ubuntu22.04-dtk26.04-0409-modelzoo`
- **DTK 版本：DTK-26.04-rc4**（文件 `/opt/dtk/.dtk_version`）
- 支持架构 `AMDGPU_TARGETS="gfx906;gfx926;gfx928;gfx936;gfx938"`
- hipcc 后端 = `/opt/dtk/dcc/bin/dcc`（clang 17.0.0，海光实现）
- **不同容器/镜像的 DTK 版本可能不同**，以目标容器内 `/opt/dtk/.dtk_version` 为准。本 skill 是结构范式，具体版本号/库版本随镜像变化。

**第二基线（2026-08-28 全量扫描）**：`<device-A> 设备（BW150 8 卡，64GB/卡，80CU/卡，1400MHz）`，
镜像 `<internal_registry>/jenkins/model_test_env/sglang:0.5.12-ubuntu22.04-dtk2604-py3.10-20260824-0006-bw150`（当时最新 dtk2604 镜像），
DTK-26.04-rc4（构建 2026-03-18），HIP 6.3.26113（`hip/bin/.hipVersion`），MIOpen 2.18.0，DUSHMEM v3.2.5，
`rocminfo` Runtime 1.11，`clinfo` OpenCL 2.1s C-3000-APP（8 卡可见）。
与第一基线同 DTK 版本，目录结构一致；本 skill 新增章节（aillvm/aicc、rocprofiler、dushmem、带宽测试工具等）均来自此基线实测。

**第三基线（2026-09-15）**：`<device-D> 设备（BW1100，8 卡，HYGON DCU-3G，gfx938）`，**物理机直接扫 /opt/dtk（非容器）**。
**DTK-25.04.4**（与 26.04 系列不同！）。**关键结构差异**（与 26.04 基线对比）：
- **无 `aillvm/` 目录**（第二套 LLVM 工具链缺失，aicc 相关不可用）；
- **无 `dushmem/` 目录**（DCU 共享内存库缺失）；
- **`AMDGPU_TARGETS` 为空**（未预设）；`/opt/dtk/bin/rocm_agent_enumerator` 只列到 `gfx936`（**未含 gfx938**，但 rocminfo 实测 gfx938、hipcc --offload-arch=gfx938 可编译）；
- `/opt/dtk/bin/` 直接有 `hipprof`/`hipprof_ctrl`/`extractkernel`/`MIOpenDriver`/`BandwidthTest`/`rocm_agent_enumerator`（26.04 基线里 hipprof 在 hip/bin，此处 bin 直放）;
- `dcc/bin/dcc` = clang 17.0.0；`rocm-smi` → `hy-smi-wrapper` 软链正确；
- 有 `cuda/`、`dcc/` 目录；`rocprofiler/bin/` 有 rocprof/rocprofv2/rocsys。
**说明**：DTK-25.04.4 是 <device-alias> **宿主机**的版本，目录比 26.04-rc4 精简（少了 aillvm/dushmem），工具路径也有差异。
> ⚠️ **重要**：**BW 系列（含 BW1100）必须用 DTK-26.04 及以上版本**，**宿主 DTK-25.04 不满足要求**。本第三基线仅描述 <device-alias> 宿主机 `/opt/dtk` 的结构（供对比参考），**不代表 BW1100 的推荐环境**——BW1100 实际编译/推理请进 **DTK-26.04 容器**（如 `pytorch:2.7.1-kylin10sp3-dtk26.04-py3.10` 内 `/opt/dtk`=DTK-26.04-rc4），勿用宿主 DTK-25.04。用本 skill 时须按实际 `/opt/dtk/.dtk_version` + 目录 presence 判断版本与能力，且 BW 系列优先用容器 DTK-26.04。

**第四基线（2026-10-09，DTK-26.10 官方包解包核对）**：官方 `DTK-26.10-Kylinv10SP3-x86_64.tar.gz`（3.28GB，md5 校验通过）全量解包核对（**非真机/容器环境，未实跑编译/框架**）。`.dtk_version`=DTK-26.10-release，rocm_version=26.10。**与 26.04-rc4 基线的关键结构差异**：
- **`aillvm/` 移除**（LLVM 小型化），`dcc/` 2.5G→749M；`dcc/bin/` 新增 `gcvm-link`/`gcvm-llc`/`gcvm-opt`（GCVM 前后端分离，独立中后端）
- **新增顶层目录**：`origami/`（liborigami.so，hipblaslt-bench 依赖它）、`lib64/`（libamd_comgr.so.2.6.0、libhsakmt.a）、`hipprofiler/`（**hipprof 从 hip/bin 移到 hipprofiler/bin**）
- **`AMDGPU_TARGETS="gfx906;gfx926;gfx928;gfx936;gfx938;gfx92a"`**（新增 gfx92a，Release Note 未提）
- **RCCL 2.27.7**（compiled with ROCm 7.2.3.0；Release Note 写 2.27.3，包内更新）
- HIP 7.2.26365（HIP_VERSION_MAJOR=7 MINOR=2），CUDA 兼容层 12.9（`#define CUDA_VERSION 12090`）
- hipprof 新参数（二进制确认）：`--realtime-trace <0=db/1=rt/2=rt-db>`、`--orderby-totalsize`、`--pmc [default|read|write|compute|wave|util|memory]`
- `include/hipfftw.h`（FFTW 接口）、`HIPSPARSE_SPMV_CSR_ALG2=3`（spmv flat 算法）、hipDNN 对齐 cuDNN（hipdnnHandle_t）、rocsolver slasr/dlasr/clasr
- dushmem v3.2.5（与 26.04 同版本未升级）
- **rocblas kernel DB 改按类型分文件**：`lib/rocblas/library_gpu6/TensileLibrary_Type_XX_..._gfx938.dat`（244 文件）+ `TensileLibrary_lazy_gfx938.dat`（lazy 加载，GEMM 算法选择不稳定疑根源）
- **hipblaslt kernel DB 改按架构分目录**：`lib/hipblaslt/library_gfx938/` 等（26.04 是 library_bw/library_bw_nmz/library_km），路径用 `HIPBLASLT_TENSILE_LIBPATH` 指定
- **GEMM 真机实测（<device-alias> BW1100，详见 dcu-dtk §5.5）**：方阵持平，+40~60%/+150% 未复现；26.10 heuristic 算法选择不稳定（rocblas 波动 2.5 倍）/选坏 kernel（hipblasLt 106k vs 最佳 270k）
- 驱动：官方要求 rock-6.4.2-V2.1.0，实测 HIP runtime 在 6.4.0-V1.7.0 上可正常初始化（探针）

## 1. 激活（容器内必做）

```bash
source /opt/dtk/env.sh     # 注入 PATH / LD_LIBRARY_PATH / CMAKE_PREFIX_PATH
```

env.sh 关键点：
- `HYHAL_PATH=/opt/hyhal`（驱动运行时，宿主机挂载进容器）
- `ROCM_PATH=/opt/dtk`，`HIP_PATH=$ROCM_PATH/hip`
- `AMDGPU_TARGETS` 见上（支持的 gfx）
- `MIOPEN_FIND_MODE=3`
- PATH 顺序：`bin:llvm/bin:hip/bin:hip/bin/hipify:hyhal/bin:opencl/bin`
- 另有 `modulefile`（Lmod 版，文案滞后写 DTK-24.04，实际以 `.dtk_version` 为准）
- `hy-smi` / `rocm-smi` 不需要 source DTK（`/opt/hyhal/bin/hy-smi` 直接可用）

## 2. 顶层结构（du 已排序）

```
/opt/dtk
├── .dtk_version  env.sh  env.csh  modulefile   # 版本 + 激活脚本
├── bin/                  # 27 文件，多为到 hip/bin、llvm/bin 的软链
├── llvm -> dcc          # 编译器主干（clang/flang/lld）
├── amdgcn -> dcc/dccgcn # device bitcode / 架构库
├── hip/                  # HIP 运行时/编译器（bin include lib cmake samples）
├── hipblas hipblaslt hipcub hipdnn hipfft hiprand hipsolver hipsparse
├── hsa/ miopen/ rccl/ rocalution/ rocblas/ rocfft/ rocprim/ rocrand/
├── rocsolver/ rocsparse/ rocthrust/ opencl/ rocprofiler/ roctracer/ hipprof_utils/
├── dushmem/              # DCU 共享内存库（NVSHMEM 对应，v3.2.5）
├── cuda/                 # CUDA 兼容层（cuda/cuda-12 + env.sh + unset_cuda_env.sh）
├── dcc/                  # 海光编译器集合（dcc/flang/lld/llvm-*/bitcode，2.5G）
├── aillvm/               # 第二套 LLVM 工具链（clang-18 系，aicc 后端，1.9G）
├── .hyhal/               # DTK 内置 hyhal 回退（正常不触发；bin 里只有 hy-smi）
├── include/  lib/  share/  src/  .info/
```

**du 排序（<device-A> 基线）**：lib 9.2G > dcc 2.5G > aillvm 1.9G > hipfft 1.2G > share 883M > cuda 377M > rocfft 346M > bin 235M > hip 44M > include 39M。
`lib/` 下 119 个 .so，含每架构预编译 kernel（`hipkernel.bin.gfx906/926/928/936/938` + `kernel_asm.hipfb`/`kernel_hip.hipfb`）。
`src/hipfor/` 是 HIP Fortran 绑定源码（hipfort*.f）；`.info/` 有 `rocm_version`（26.04）/`version-{dev,libs,utils}`/`md5-list`。

关键软链：
| 软链 | 指向 | 说明 |
|---|---|---|
| `/opt/dtk/llvm` | `dcc` | DTK 编译器主干 |
| `/opt/dtk/amdgcn` | `dcc/dccgcn` | device bitcode/架构库 |
| `/opt/dtk/bin/hipcc` | `hip/bin/hipcc` → `../llvm/bin/dcc` | HIP 编译器实为 dcc |
| `/opt/dtk/bin/rocminfo` | `hip/bin/rocminfo` | 设备枚举 |
| `/opt/dtk/bin/rocm-smi` | `hy-smi-wrapper` | = hy-smi 包装 |
| `/opt/dtk/lib/libamdhip64.so.{4,5}` | `libgalaxyhip.so` | HIP runtime 后端是 galaxyhip |

## 3. 工具脚本总览（已逐个读源码）

### 3.1 bin 直放
| 工具 | 类型 | 用途 |
|---|---|---|
| `clang-ocl` | bash | OpenCL .cl 离线编译成 code object（拼 opencl/ocml/ockl/oclc_* bitcode） |
| `hipgdb` | bash | 按驱动版本自动选 amd-dbgapi(0.56/0.70/0.77) 的 HIP 调试器；支持 `HIPGDB_FORCE_USE_VERSION`；`bin/hipgdb-11/13/15/` 是按 dbgapi 版本分目录的实体 |
| `install_precompiled_kernels.sh` | bash | 按 rocminfo 探测 arch-CU 装 MIOpen kdb 预编译库 |
| `test_perf.py` | python3 | MIOpen 性能回归/新旧对比脚本（±5% 容差） |
| `aicc` / `aicc.pl` | ELF/perl | **AICC 编译器**（clang-18 系，后端在 `aillvm/`，`aicc --version` 报 HIP 6.2.0 + clang 18.0.0）；`aicc.pl` 是 perl 包装（解析 `--rocm-path/--hip-path`，`HIPCC_VERBOSE` 0x1 命令/0x2 路径/0x4 aicc 参数，`HIPCC_COMPILE_FLAGS_APPEND`/`HIPCC_LINK_FLAGS_APPEND` 追加参数）。**FA 自编译等场景的"无 aicc 命令→装 AICC（dtk_llvm.run）"坑，装好后就是它** |
| `aiconfig` / `aiconfig.pl` / `aivars.pm` | ELF/perl | aicc 配套配置查询（= hipconfig 的 aicc 版：`-p/-R/-C/-c/-P/-r/-l/-f/-v/--check`） |
| `BandwidthTest` | ELF | **卡间 P2P 带宽/延迟测试**（<device-A> 实测 `-l` 列 8 卡：`Name=BW150 Mem=64.0GB #CUs=80 Freq=1400Mhz`）：`-l` 列设备、`-a` 全卡单向带宽、`-L` 单向延迟、`-A` 双向带宽、`-R` 开 P2P 读（默认写） |
| `rocm-bandwidth-test` | ELF | ROCm 系带宽测试（v2.6.0）：`-e` 列设备、`-t` 拓扑+可分配内存、`-l` 延迟、`-m <MB...>` 指定 buffer 大小、`-b` 双向拷贝设备对、`-v` 校验模式 |
| `MIOpenDriver` | ELF | **MIOpen 算子级测试/调优**（<device-A> 实测）：base arg 支持 `conv[fp16\|int8\|bfp16\|fp8\|bfp8]`、`gemm[fp16]`、`pool/lrn/activ/softmax/bnorm/rnn/ctc/dropout/tensorop/tensor_trans/reduce[fp16\|fp64]`、`layernorm/sum/groupnorm/cat/addlayernorm/getitem/rope[fp16\|bfp16]`、`mha[fp16\|bfp16]`、`fusion[fp16\|int8]`、`adam/ampadam/transformersadamw/transformersampadamw` 等——**fp8/bf8 卷积算子验证用它** |
| `hy-smi-wrapper` / `rocm-smi` | ELF/软链 | `rocm-smi` → `hy-smi-wrapper` → 实际调 `/opt/hyhal/bin/hy-smi`（ROCm 命令兼容层） |
| `dushmem-info` | ELF | DUSHMEM（DCU 共享内存库，NVSHMEM 对应）信息查询：`-a` 全部、`-n` 版本（<device-A> 实测 v3.2.5）、`-b` 构建信息、`-e` 环境变量、`-d` 含隐藏变量、`-r` RST 格式 |

### 3.2 hip/bin
| 工具 | 类型 | 用途 |
|---|---|---|
| `hipcc` | 软链→dcc | HIP 编译器 |
| `hipconfig` | perl | 打印 HIP 配置（path/compiler/platform/cpp_config/version/full/check） |
| `hipvars.pm` | perl | HIP 环境变量解析模块（被 hipconfig/hipcc/extractkernel 用） |
| `extractkernel` | perl | 从可执行解 offload bundle，反汇编出各 gfx 的 .hsaco / .isa |
| `hip_embed_pch.sh` | bash | 生成 HIP PCH / RTC 头（构建期用） |
| `rocm_agent_enumerator` | bash | 列出支持 gfx（gfx000/906/926/928/936/938） |
| `hipcc_cmake_linker_helper` | bash | CMake 链接辅助 |
| `hipify/` | 目录 | CUDA→HIP 移植工具集（hipify-perl 等） |
| `hiplogdump` `hipprof` `hipprof_ctrl` | ELF | HIP 性能剖析 |
| `rocminfo` | ELF | 枚举 DCU 设备/agent/CU/BDFID |

### 3.3 dcc/bin（clang-17 系）
`dcc` `clang` `clang++` `clang-cl` `flang` `lld` `llc` `opt` `llvm-ar` `llvm-link`
`llvm-objcopy` `llvm-readelf` `llvm-config` `clang-offload-bundler` `amdgpu-arch`
`dccobjdump` `dcc-gicc` `dccturing` `llvm-omp-*` 以及 **`gpurun`**（详见下）。

#### gpurun（最常用 NUMA 亲核脚本，375 行）
用 `taskset` 把 GPU 进程绑到与所选 GPU **同一 NUMA 域的 CPU 核**，提升访存效率；
多 MPI rank 共享 GPU 时用 `HSA_CU_MASK` 把 CU 互不重叠分给各 rank；导出
`ROCR_VISIBLE_DEVICES` / `OMPX_TARGET_TEAM_PROCS` / `HSA_CU_MASK`。
读 OpenMPI 的 `OMPI_COMM_WORLD_LOCAL_*` 或 Platform MPI / MPI_COMM_WORLD 变量定 rank；
找不到则单进程单卡。环境变量：`GPURUN_VERBOSE`(空/0/1/2)、`ROCMINFO_BINARY`、`AOMP`、`GPURUN_DEVICE_BIAS`。

### 3.4 aillvm/bin（clang-18 系，aicc 后端，<device-A> 实测）
DTK 内**第二套 LLVM 工具链**（`/opt/dtk/aillvm/`，1.9G），是 `aicc` 编译器的后端（对应 dcc 之于 hipcc）。
`aillvm/bin` 内容：`clang-18` `clang++-18` `clang-format` `clang-tidy` `clangd` `diagtool`（诊断工具）
`hmaptool`（hash map 工具）`dsymutil`（dSYM 符号）`amdgpu-offload-arch` `clang-nvlink-wrapper`
`llvm-ar/llvm-link/llvm-objcopy/llvm-readelf/llvm-config/opt/llc/lld` 等标准 LLVM 工具集。
注意：`aillvm/bin` **不在默认 PATH**（env.sh 只加 `llvm/bin` 即 dcc），要用绝对路径或自行 export。
`aicc --version` 报 HIP 6.2.0 + clang 18.0.0（与 hipcc 的 clang-17 并存，两套编译器独立）。

### 3.5 rocprofiler/bin（性能剖析三件套，<device-A> 实测）
`/opt/dtk/rocprofiler/bin/` 有 `rocprof`（RPL 版）、`rocprofv2`、`rocsys`。
**关键坑：该目录不在默认 PATH**（env.sh 的 PATH 里没有 `rocprofiler/bin`），
`source /opt/dtk/env.sh` 后直接敲 `rocprof` 会报 command not found，必须用绝对路径
`/opt/dtk/rocprofiler/bin/rocprof`（或自行 export PATH）。

> **BW1000 容器补充（2026-09-10）**：`/opt/dtk/bin/rocminfo` 在 BW1000 容器（torch 2.10.0+das.opt1.dtk2604 镜像，hyhal 6.3.30）每次运行都打
> `WARNING: /opt/hyhal/lib/cmake/rocm_smi doesn't exist, please install the latest driver(version >= 6.3.15)` + `The rocm_smi_lib will be used in default`，
> 随后正常输出（gfx936/80CU/64GB 可查）——**纯警告无实际影响**，hy-smi 正常。脚本里过滤这两行即可。

| 工具 | 用途 | 关键参数（<device-A> 实测 --help） |
|------|------|------|
| `rocprof` | RPL 性能剖析（计数器采集） | `--list-basic`/`--list-derived` 列计数器、`-i <input .txt/.xml>` 指定计数器输入、`-o <output CSV>` 输出、后接 `<app>` 被剖析程序；计数器定义在 `/opt/dtk/rocprofiler/lib/rocprofiler/metrics.xml` |
| `rocprofv2` | 新一代剖析器（trace 为主） | `--list-counters`、`--hip-api`/`--hip-activity`/`--hip-trace`、`--hsa-api`/`--hsa-activity`、`--roctx-trace`、`--kernel-trace`、`--sys-trace`、`--plugin [file/perfetto/att/ctf]`、`-i pmc.txt`、`-o`、`-d <out_dir>` |
| `rocsys` | 系统级剖析会话（可挂 MPI） | `rocsys --session <name> start/stop`；`rocsys --session-new <name> launch py app.py`（MPI 场景 mpiexec 参数放 rocsys 之前） |

配套：`/opt/dtk/share/profiler/counters/` 有 `basic_counters.xml`/`derived_counters.xml`（rocprof 计数器集）；
`lib/` 有 `librocprofiler64(.1/.2/v2)` `libroctracer64` `libroctx64` `libperfetto`。
`hipprof`/`hipprof_ctrl`（hip/bin，在 PATH 里）是 HIP 层剖析，与 rocprof 互补——详见 dcu-perf-profiling。

### 3.6 dushmem（DCU 共享内存库，NVSHMEM 对应，<device-A> 实测）
`/opt/dtk/dushmem/`（v3.2.5）：`lib/` 含 `libdushmem_host.so.3.2.5`、`libdushmem_device.a`、
bootstrap 库 `dushmem_bootstrap_{mpi,pmi,pmi2,uid}.so`、传输库 `dushmem_transport_{ibgda,ibrc}.so`
（IBGDA=GPUDirect RDMA 类、IBRC=RC 传输，对应 NVSHMEM 的 IBGDA/IBRC transport）。
`bin/dushmem-info` 查版本/构建/环境变量（见 3.1）。`env.sh` 已把 `dushmem/lib` 加进 LD_LIBRARY_PATH。
多机 GPU 单边通信（put/get/signal）场景用它，RCCL 是集合通信、DUSHMEM 是点对点单边，两者互补。

## 4. lib 库分组（119 个 .so 摘要）
- HIP 运行时：`libamdhip64.so`(→galaxyhip) `libhiprtc.so` `libopenmphip.so`
- 数学/线性代数：`librocblas` `libhipblas` `libhipblaslt` `libhipsolver` `librocsolver` `librocsparse` `libhipsparse` `librocfft` `libhipfft` `libhipfftMp`
- 随机/通信：`libhiprand` `librocrand` `librccl`(1.0) `libnccl_net.h`
- DNN/图像：`libMIOpen`(+recommend) `libhipdnn`(+backend) `librpp` `libduimgcodec`
- 运行时/剖析/调试：`libhsa-runtime64` `libamd_comgr` `librocprofiler64(.1/.2/v2)` `libroctracer64` `libroctx64` `libhsa-amd-aqlprofile64` `librocm-dbgapi`(0.56/0.70/0.77) `librocm-debug-agent` `libperfetto` `libhipprof_utils`
- 辅助：`liblz4` `libnanomsg` `libunwind` `libleak_check` `libtrace_lib_call` `mpitracer` `librocm-core`

## 5. 其他重要目录
- `cuda/`：CUDA 兼容层，`source /opt/dtk/cuda/env.sh` 启用，`unset_cuda_env.sh` 撤销
- `hip/samples/`：HIP 示例（0_Simple 1_Utilities 2_Advanced 3_HIPLibraries）
- `share/`：`doc/`(各库 API 文档) `miopen/`(db+model) `rccl/`(msccl 算法) `profiler/counters`(rocprof 计数器 xml) `pkgconfig/`
- `dcc/dccgcn/bitcode/`：`asanrtl.bc cuda2gcn.bc hip.bc ockl_base.bc` 及每架构 `hyptxas-device-gfx90x.bc`
- `.info/`：`rocm_version` `version-{dev,libs,utils}` `md5-list`（全量 md5 校验）

## 6. 常用命令速查
```bash
source /opt/dtk/env.sh
hipcc main.hip -o main            # HIP 编译（实际走 dcc/clang-17）
hipconfig --full                 # 完整 HIP/DTK 配置
rocminfo | grep -E 'gfx|Compute Unit|Name'
rocm_agent_enumerator           # 支持 gfx 列表
extractkernel -i ./a.out        # 抽取反汇编设备 kernel
clang-ocl -mcpu=gfx906 -o k.o kernel.cl
gpurun ./my_gpu_app             # NUMA 亲核启动（多 rank 自动 CU 掩码）
hipgdb ./a.out                  # HIP 调试（自动选 dbgapi）
/opt/dtk/rocprofiler/bin/rocprof ./a.out   # 性能剖析（rocprof 不在默认 PATH，用绝对路径）
BandwidthTest -l                # 列卡；-a 单向带宽 / -L 延迟 / -A 双向
MIOpenDriver conv[fp16] ...     # MIOpen 算子级测试（fp8/bf8 卷积验证）
source /opt/dtk/cuda/env.sh     # 启用 CUDA 兼容环境
```

## 6.5 驱动侧（/opt/hyhal）编译/PTX 工具链（<device-A> 实测）

> 来源：2026-08-28 在 <device-A> 设备对 `/opt/hyhal/bin` 扫描 + `--help` 实测。DTK 的 hipcc/dcc 之外，
> **驱动目录自带一套 PTX/设备码工具**，离线处理 code object / fat binary 时可用（不依赖 DTK 激活，
> 但 `hycc` 需要 `HIP_CLANG_PATH` 指向 clang）：

| 工具 | 用途 | 关键参数 |
|------|------|------|
| `hyptxas`（v1.0.5） | **PTX 汇编器**（对应 ROCm 的 ptxas） | `-offload-arch <gfx>`、`-o <file>`、`-c` 仅编译、`-compile-llvmir/-emit-llvmir`、`-clang-path/-rocm-path/-device-lib-path` |
| `hycc` | HIP C 前端编译器 | 需 `export HIP_CLANG_PATH=<clang路径>`，裸跑报 `missing clang path` |
| `hyfatbinary` | **fat binary 生成**（多架构 code object 打包） | `-create <file>`、`-image3 <file,kind,sm...>`、`-offload-arch`、`-link`、`-c` device-c、`-fatbin` 内嵌 C 头 |
| `extractkernel`（DTK hip/bin） | 从可执行解 offload bundle 反汇编（见 3.2） | 与 hyfatbinary 互补：一个拆一个打包 |

配套资源：`/opt/hyhal/bitcode/` 有 `hyptxas-device-gfx906/926/928.bc`（PTX 汇编器 device 库，
`-device-lib-path` 指向处）；`/opt/hyhal/env.sh` 会把 `hyhal/bin` 加进 PATH（DTK env.sh 的 PATH 里也含 `hyhal/bin`）。

> 正常开发流程仍走 DTK `hipcc`（见第 6 节速查）；hyptxas/hyfatbinary 主要用于**离线处理/调试 code object**
> （如单独重汇编某 gfx 的 PTX、手工拼 fat binary 排查多架构链接问题）。

## 7. 与其他 skill 衔接
- **dcu-dtk**：安装/激活规则（仅物理机）；本 skill 是容器内的结构速查
- **dcu-docker**：容器自带 DTK，不要容器内重装；用 `source /opt/dtk/env.sh`
- **dcu-env**：编译/运行前确认 `which hipcc` / `echo $DTK_PATH` / `echo $AMDGPU_TARGETS`
- **dcu-cpu-affinity**：`gpurun` 是 DTK 内置 NUMA 绑核+CU 掩码；手动拓扑优化看该 skill
- **dcu-hy-smi**：`hy-smi`/`rocm-smi` 来自 hyhal，不依赖 DTK 激活
- **dcu-ontrack-experience**：遇到 libxxx.so 缺失/编译失败/版本冲突，按现象反查修复

## References（本 skill 自带，已读源码）
- 详细目录与工具文档：`references/dtk-layout-and-tools.md`（含每个脚本源码要点、gpurun 详解、lib 库全列表）
- 工具脚本源码库（原样保存，未改动）：`references/dtk-scripts/`
  - `clang-ocl` `hipgdb` `install_precompiled_kernels.sh` `test_perf.py`
  - `extractkernel` `hipconfig` `hipvars.pm` `hip_embed_pch.sh` `hipcc_cmake_linker_helper` `rocm_agent_enumerator`
  - `gpurun`（375 行 NUMA 亲核脚本）
