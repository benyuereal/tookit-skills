# DTK 目录结构与工具脚本全解（实测于 <container_vllm015> 容器）

> 来源：设备 `<device_name>`，容器 `<container_vllm015>`
> 镜像：`<internal_registry>/dcu/admin/base/custom:vllm015-ubuntu22.04-dtk26.04-0409-modelzoo`
> 实测 DTK 版本：**DTK-26.04-rc4**（文件 `/opt/dtk/.dtk_version`）
> 容器内路径：`/opt/dtk`（`/opt/dtk` 即 `ROCM_PATH` / `DTKROOT`）
> 采集日期：2026-08-01

> 用途：本文件是 `dcu-dtk` 的“目录与工具速查”补充。安装/激活规则看主 SKILL.md；
> 本文件讲清楚 DTK 内部有什么、每个工具脚本干什么、源码摘要，便于日常排查和二次开发。

---

## 0. 激活与环境变量（容器内，必读）

```bash
source /opt/dtk/env.sh        # 标准激活；注入 PATH/LD_LIBRARY_PATH/CMAKE_PREFIX_PATH
```

env.sh 关键点（实测）：
- `export HYHAL_PATH=/opt/hyhal` —— DCU 驱动运行时（宿主机挂载进容器，见 dcu-docker）
- `export ROCM_PATH=$DTKROOT`（`/opt/dtk`）
- `export HIP_PATH=${ROCM_PATH}/hip`
- `export AMDGPU_TARGETS="gfx906;gfx926;gfx928;gfx936;gfx938"` —— **当前 DTK 支持的 DCU/GCN 架构**
- `export MIOPEN_FIND_MODE=3`
- 若 `/opt/hyhal/lib` 不存在，则回退用 DTK 自带 `.hyhal`（容器自带时一般不会触发）
- PATH 注入（顺序）：`$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$ROCM_PATH/hip/bin:$ROCM_PATH/hip/bin/hipify:$HYHAL_PATH/bin:$ROCM_PATH/opencl/bin`
- LD_LIBRARY_PATH 注入：`lib lib64（DTK）+ $HYHAL_PATH/lib lib64 + dushmem/lib + opencl/lib + dcc/gcvm/lib + hip/lib + llvm/lib`
- 另有 `modulefile`（Lmod 版，内容同 env.sh 的变量），供 `module load dcu` 体系使用；其 whatis 文案仍写 “DTK-24.04”，**文案滞后，实际版本以 `.dtk_version` 为准**。

> 容器内 hy-smi 不需要 source DTK（`/opt/hyhal/bin/hy-smi` 直接可用）。
> 但 hipcc / rocminfo / rocprof / 编译等需 source 后才有。

---

## 1. 顶层目录结构（du 已排序，单位约）

```
/opt/dtk
├── .dtk_version          # 版本标记：DTK-26.04-rc4
├── env.sh  env.csh  modulefile   # 激活脚本（bash/csh/Lmod）
├── bin/                  # 27 个文件，多为到 hip/bin、llvm/bin 的软链（232M? 实际 bin 自身小，多数软链）
├── llvm -> dcc           # 软链：llvm 就是 dcc 编译器目录
├── amdgcn -> dcc/dccgcn  # 软链：bitcode/device lib 目录
├── hip/                  # HIP 运行时/编译器（232M），含 bin/ include/ lib/ cmake/ samples/
├── hipblas hipblaslt hipcub hipdnn hipfft hiprand hipsolver hipsparse
├── hsa/                  # HSA 运行时（12K，thin wrapper）
├── miopen/ rccl/ rocalution/ rocblas/ rocfft/ rocprim/ rocrand/ rocsolver/ rocsparse/ rocthrust/
├── opencl/               # OpenCL 运行时（3.2M）
├── rocprofiler/          # 性能剖析（8.2M）
├── roctracer/            # 追踪（524K）
├── hipprof_utils/        # hip 性能剖析工具（7.0M）
├── dushmem/              # DCU 共享内存库（8K）
├── cuda/                 # CUDA 兼容层（377M）：cuda/cuda-12 + env.sh + unset_cuda_env.sh
├── dcc/                  # 海光编译器集合（2.5G）：dcc(clang) / flang / lld / llvm-* / bitcode
├── .hyhal/               # DTK 内置 hyhal 回退（容器自带 hyhal 时基本不触发）
├── include/              # 全部头文件（39M）：hip/ rocblas.h rocsolver.h ... 等
├── lib/                  # 运行时库（9.2G）：136 个条目，119 个 .so
├── share/               # 文档/DB/pkgconfig（883M）：doc/ miopen/ rccl/ profiler/ rocm/ pkgconfig
├── src/                  # 部分源码（4.2M）
└── .info/                # 版本与 md5 清单（rocm_version / version-* / md5-list 1.4M）
```

### 关键软链
| 软链 | 指向 | 说明 |
|---|---|---|
| `/opt/dtk/llvm` | `dcc` | DTK 编译器主干（clang/flang/lld） |
| `/opt/dtk/amdgcn` | `dcc/dccgcn` | device bitcode / 架构相关库 |
| `/opt/dtk/bin/hipcc` | `hip/bin/hipcc` → `../llvm/bin/dcc` | HIP 编译器实际是 dcc（clang 派生） |
| `/opt/dtk/bin/rocminfo` | `hip/bin/rocminfo` | 设备枚举 |
| `/opt/dtk/bin/rocm-smi` | `hy-smi-wrapper` | 等于 hy-smi 包装 |
| `/opt/dtk/lib/libamdhip64.so.{4,5}` | `libgalaxyhip.so` | HIP runtime 实际后端是 galaxyhip（海光实现） |

---

## 2. 工具脚本总览（bin 层 + hip/bin + dcc/bin）

`/opt/dtk/bin` 下 27 个文件，**绝大多数是软链**，真正落地在：
- `/opt/dtk/hip/bin/` —— HIP 工具（脚本/perl/elf）
- `/opt/dtk/dcc/bin/` —— 编译器工具（clang/flang/lld/llvm-*、gpurun）
- `/opt/dtk/bin/` 直接放的几个 shell/python（clang-ocl、hipgdb、install_precompiled_kernels.sh、test_perf.py）

### 2.1 bin 直放（脚本类，已读源码）

| 工具 | 类型 | 用途 | 源码要点 |
|---|---|---|---|
| `clang-ocl` | bash (107行) | **OpenCL 内核离线编译**。把 `.cl` 编译成 code object（.hsaco/可执行 bundle）。 | 解析 `-o/-mcpu=gfx*/-mcode-object-version`；默认 gfx803、codeobj v4；拼 `opencl/ocml/ockl/oclc_*` bitcode 后用 `clang -emit-llvm + llvm-link + clang` 生成；`grep warning` 失败即退出。bitcode 目录取 `$ROCM_PATH/amdgcn/bitcode` 或 `$ROCM_PATH/lib/bitcode`。 |
| `hipgdb` | bash (101行) | **HIP 调试器启动器**（基于 amd-dbgapi + gdb）。按驱动版本选 hipgdb/dbgapi 版本。 | 读 `/sys/module/hycu/version` 或 `hydcu` 或 `amdgpu` 定驱动版本；≥6.3 用 v15(amd-dbgapi 0.77)，5.7–6.2 用 v13(0.70)，否则 v11(0.56)；也可 `HIPGDB_FORCE_USE_VERSION=hyhal\|15\|13\|11\|new\|old` 强制；`HIPGDB_DRIVER_VERSION_CHECK=1` 只打印不执行。 |
| `install_precompiled_kernels.sh` | bash (39行) | **安装 MIOpen 预编译 kernel 数据库（kdb）**。按 rocminfo 探测到的 arch-CU 组合装对应 `miopen-hip-<arch>-<cu>kdb` 包。 | 调 `rocminfo` 抓 `gfx*/Compute Unit:`；rhel 走 `yum`，ubuntu 走 `apt`；包名 `miopen-${backend}-${arch}-${num_cu}kdb`。 |
| `test_perf.py` | python3 (261行) | **MIOpen 性能回归测试/对比脚本**。读取 perf_models 下的模型文件，批量跑 MIOpenDriver 并解析 kernel time，可对比新旧结果（±5% 容差）。 | 参数 `--filename --install_path --compare_results --old_results_path --override`；`run_driver_cmds` 拼 `MIOpenDriver -V 0 -i 10 -w 1 -t 1 -G 1` 并抓 `GPU Kernel Time Elapsed`；`compare_file` 按 `k_time` 算 speedup，`< -5%` 判 FAILED。结果写 `<脚本目录>/perf_results/`。 |

### 2.2 hip/bin 工具（已读源码）

| 工具 | 类型 | 用途 | 源码要点 |
|---|---|---|---|
| `hipcc` | 软链 → `../llvm/bin/dcc` | **HIP 编译器**（实为 dcc/clang）。 | 见 dcc。 |
| `hipconfig` | perl (246行) | **打印 HIP 配置**：path/rocmpath/compiler/platform/runtime/cpp_config/version/full/check。 | 依赖 `hipvars.pm`；clang 模式下 `CPP_CONFIG` 注入 `-D__HIP_PLATFORM_AMD__ -I hip/include -I clang/$ver`；`--full` 会跑 hipcc --cxxflags/--ldflags，`--check` 校验 hipconfig 在 PATH 且 LD_LIBRARY_PATH 含 HSA_PATH。 |
| `hipvars.pm` | perl (152行) | **HIP 环境变量解析模块**（被 hipconfig/hipcc/extractkernel 用）。 | 推导 `HIP_PATH/ROCM_PATH/HSA_PATH/HIP_CLANG_PATH/HIP_PLATFORM/HIP_COMPILER/HIP_RUNTIME/HIP_VERSION`；读 `$HIP_PATH/lib/.hipInfo` 与 `bin/.hipVersion`；优先 env，其次 .hipInfo。默认 platform=amd、compiler=clang、runtime=rocclr。 |
| `extractkernel` | perl (258行) | **从可执行文件抽取设备端 kernel**：解 offload bundle，生成各 gfx 的 `.hsaco` 并用 `llvm-objdump` 反汇编出 `.isa`。 | 解析 `.kernel`(hcc) 或 `.hip_fatbin`(clang) 段；用 `clang-offload-bundler -unbundle` 拆出每架构 hsaco；`llvm-objdump --disassemble --mcpu=<arch>` 生成 ISA。调试变量 `$debug`。 |
| `hip_embed_pch.sh` | bash (219行) | **生成 HIP 预编译头（PCH）/RTC 头**（构建期用）。 | `generate_pch`：用 clang 生成 wave32/wave64 的 `.pch` 并 `.incbin` 进 `hip_pch.o`；`generate_rtc`：生成 `libhiprtc-builtins.so` 与 hiprtc_header.o。参数：`HIP_BUILD_INC_DIR HIP_INC_DIR HIP_AMD_INC_DIR LLVM_DIR [-p|-r] [RTC_LIB_OUTPUT]`。 |
| `rocm_agent_enumerator` | bash (7行) | **列出支持的 gfx 架构**（给 hipcc/编译用）。 | 直接 echo：`gfx000 gfx906 gfx926 gfx928 gfx936 gfx938`。 |
| `hipcc_cmake_linker_helper` | bash (30行) | **CMake 链接辅助**：根据 hipconfig --compiler 决定把第一个参数当 HIP_CLANG_PATH/HCC_HOME 传给 hipcc。 | 调 `hipconfig --compiler`：hcc→`HCC_HOME=$1`，clang→`HIP_CLANG_PATH=$1`，否则原样。 |
| `hipify/` | 目录 | **CUDA→HIP 代码移植工具集**（perl 脚本）。 | 含 `hipify-perl hipconvertinplace-perl.sh hipexamine-perl.sh findcode.sh finduncodep.sh hipify-cmakefile`。 |
| `hiplogdump` `hipprof` `hipprof_ctrl` | ELF | HIP 性能剖析相关可执行（hipprof_utils 提供）。 | 二进制，未读源码。 |
| `hipvars.pm` / `target.lst` | perl / txt | 配置与架构清单。 | `target.lst` 内容即 `gfx906 gfx926 gfx928 gfx936 gfx938`。 |
| `rocminfo` | ELF | **枚举 DCU 设备/agent/CU/BDFID**（兼容性探测核心工具）。 | 二进制。 |

### 2.3 dcc/bin 编译器工具

`/opt/dtk/dcc/bin`（llvm 系，版本 **clang-17**）：

| 工具 | 类型 | 用途 |
|---|---|---|
| `dcc` | ELF | **海光 HIP/GCN 编译器**（= clang，HIP 编译后端） |
| `clang` `clang++` `clang-cl` `clang-cpp` | 软链→clang-17 | C/C++/CUDA/HIP 前端 |
| `clang-17` | ELF | clang 17 本体 |
| `flang` `flang1` `flang2` | ELF | **Fortran 编译器**（LLVM flang） |
| `lld` `ld.lld` `ld64.lld` `lld-link` | ELF/软链 | 链接器 |
| `llc` `opt` | ELF | LLVM 后端代码生成 / 优化 |
| `llvm-ar` `llvm-link` `llvm-objcopy` `llvm-readelf` `llvm-readobj` `llvm-cxxfilt` | ELF | LLVM 工具链 |
| `llvm-config` | ELF | 打印 LLVM 编译/链接标志 |
| `llvm-lto` `llvm-lto2` | ELF | LTO |
| `clang-offload-bundler` `clang-offload-wrapper` `clang-build-select-link` `clang-linker-wrapper` | ELF | **offload 捆绑/解绑/包装**（HIP 多架构 fatbin 关键） |
| `amdgpu-arch` | ELF | 打印当前 GPU 架构 |
| `dccobjdump` | ELF | 海光版 objdump（反汇编） |
| `dcc-gicc` `dccturing` | ELF | 海光专用编译/调优工具（gicc=通用中间码编译器？turing=？） |
| `gpurun` | bash (375行) | **GPU 进程启动器（OpenMP/MPI 亲核+CU 掩码）**，详见下。 |
| `openmpi_set_cu_mask` | 软链→gpurun | 同 gpurun |
| `llvm-omp-device-info` `llvm-omp-kernel-replay` `prep-libomptarget-bc` | ELF | OpenMP target offload 辅助 |

#### gpurun 详解（dcc/bin/gpurun，375 行，最常用的 NUMA 亲核脚本）
> 作用：用 `taskset` 把 GPU 进程绑到与所选 GPU **同一 NUMA 域的 CPU 核**，提升访存效率；多 MPI rank 共享 GPU 时用 `HSA_CU_MASK` 把 CU 互不重叠地分给各 rank；导出 `ROCR_VISIBLE_DEVICES`/`OMPX_TARGET_TEAM_PROCS`/`HSA_CU_MASK`。

- 输入：OpenMPI 的 `OMPI_COMM_WORLD_LOCAL_SIZE/RANK`，或 Platform MPI 的 `MPI_LOCALNRANKS/MPI_LOCALRANKID`，或 `MPI_COMM_WORLD_LOCAL_*`，都没有则单进程单卡。
- 流程：调 `rocminfo` 抓 GPU 数、每 GPU 的 CU 数、BDFID；用 `lscpu --extended` + `/sys/bus/pci/devices/0000:<bdf>.0/numa_node` 定 NUMA 节点；算 rplace（rank 的 CU 子集）与 16 进制 `HSA_CU_MASK`；`taskset -c <同NUMA的cpu列表>` 启动应用。
- 环境变量：`GPURUN_VERBOSE`（空/0=1行 trace，1=详细，2=含 taskset 命令）、`ROCMINFO_BINARY`、`AOMP`（默认 `/opt/rocm`，找不到报错）、`GPURUN_DEVICE_BIAS`（设备偏移）。
- 输出 env：`ROCR_VISIBLE_DEVICES`（设备>1 时设）、`OMPX_TARGET_TEAM_PROCS`（可用 CU 数）、`HSA_CU_MASK`（仅多 rank 共享 GPU 时设 `0:<mask>`）。
- 限制：每进程至多 1 GPU；CU 掩码互斥（不共享）；需 AOMP 15.0/ROCm 5.0+。

> 与 dcu-cpu-affinity skill 关联：gpurun 是 DTK 自带的“按 GPU 绑 NUMA 核 + CU 掩码”方案；手动绑核/拓扑优化可参考 `dcu-cpu-affinity`。

---

## 3. 主要库（/opt/dtk/lib，119 个 .so 摘要）

运行时库按功能分组（实测 `ls lib/*.so*`）：
- **HIP 运行时**：`libamdhip64.so`（→`libgalaxyhip.so`）、`libhiprtc.so`、`libopenmphip.so`
- **数学/线性代数**：`librocblas.so`(4.3) `libhipblas.so`(2.3) `libhipblaslt.so`(0.10) `libhipsolver.so` `librocsolver.so`(0.1) `librocsparse.so`(1.1) `libhipsparse.so`(1.1) `librocfft.so`(0.1) `libhipfft.so`(0.1) `libhipfftMp.so` `libhipcub`/`librocprim`/`librocthrust`（header-only，库在 include）
- **随机/通信**：`libhiprand.so`(1.1) `librocrand.so`(1.1) `librccl.so`(1.0) `libnccl_net.h`（rccl 网络插件头）
- **DNN/图像**：`libMIOpen.so`(1.0) `libMIOpen-recommend.so` `libhipdnn.so` `libhipdnn_backend.so` `librpp.so`(6.0) `libduimgcodec.so`(0.2)
- **运行时/剖析/调试**：`libhsa-runtime64.so` `libamd_comgr.so`(2.6) `librocprofiler64.so(.1/.2/v2)` `libroctracer64.so` `libroctx64.so` `libhsa-amd-aqlprofile64.so` `librocm-dbgapi.so`(0.56/0.70/0.77) `librocm-debug-agent.so` `libperfetto.so` `libhipprof_utils.so` `libamdhip64` 等
- **辅助/工具**：`liblz4.so` `libnanomsg.so` `libunwind.so` `libleak_check.so` `libtrace_lib_call.so` `mpitracer.so` `librocm-core.so`(1.0.50300)

lib 子目录：`amd-dbgapi/ cmake/ duimgcodecExt/ dushmem/ hipblaslt/ hipdnn_plugins/ hipdnn_reference_data/ rocblas/ rocfft/`

---

## 4. 其他重要目录

- **`/opt/dtk/cuda/`**：CUDA 兼容层（cuda/cuda-12 + env.sh + unset_cuda_env.sh）。`source /opt/dtk/cuda/env.sh` 注入 CUDA 兼容环境；`unset_cuda_env.sh` 撤销。用于跑 CUDA 源码/库。
- **`/opt/dtk/hip/samples/`**：HIP 示例（`0_Simple 1_Utilities 2_Advanced 3_HIPLibraries`），移植/验证可参考。
- **`/opt/dtk/share/`**：`doc/`（各库 API 文档：hipblas/rocblas/miopen/rccl/…）、`miopen/`（db + model 性能库）、`rccl/`（msccl 算法）、`profiler/counters`（rocprof 计数器 xml）、`pkgconfig/`、`rocm/`。
- **`/opt/dtk/dcc/dccgcn/bitcode/`**：设备端 bitcode：`asanrtl.bc cuda2gcn.bc hip.bc ockl_base.bc` 及每架构 `hyptxas-device-gfx90x.bc`、`opencl.amdgcn.bc` 等（`clang-ocl` 用到）。
- **`/opt/dtk/.info/`**：`rocm_version` `version-{dev,libs,utils}` `md5-list`（全量 md5 校验清单）。
- **`/opt/dtk/.hyhal/`**：DTK 内置 hyhal 回退（含 bin/hsa/lib/rocm_smi 等）；正常情况下容器用宿主机挂载的 `/opt/hyhal`，不触发。

---

## 5. 常用命令速查（容器内，已 source DTK）

```bash
source /opt/dtk/env.sh

hipcc main.hip -o main            # HIP 编译（实际走 dcc/clang-17）
hipconfig --full                 # 看完整 HIP/DTK 配置
rocminfo | grep -E 'gfx|Compute Unit|Name'   # 看设备/架构/CU 数
rocm_agent_enumerator           # 看支持的 gfx 列表
extractkernel -i ./a.out        # 抽取并反汇编设备 kernel（生成 *.hsaco / *.isa）
clang-ocl -mcpu=gfx906 -o k.o kernel.cl   # OpenCL 离线编译
gpurun ./my_gpu_app             # NUMA 亲核启动（多 rank 时自动 CU 掩码）
hipgdb ./a.out                  # HIP 调试（按驱动版本自动选 dbgapi）
rocprof ./a.out                 # 性能剖析（rocprofiler/bin）
source /opt/dtk/cuda/env.sh     # 启用 CUDA 兼容环境
```

---

## 6. 与本包其他 skill 的衔接

- **dcu-docker**：容器自带 DTK，**不要**在容器内跑主 SKILL.md 的“下载解压装 DTK”。只需 `source /opt/dtk/env.sh`。
- **dcu-env**：编译/运行前需确认 `which hipcc`、`echo $DTK_PATH`、`echo $AMDGPU_TARGETS`。
- **dcu-cpu-affinity**：`gpurun` 是 DTK 内置的 NUMA 绑核 + CU 掩码工具；手动拓扑优化参考该 skill。
- **dcu-hy-smi**：`hy-smi` / `rocm-smi` 来自 hyhal，不依赖 DTK 激活；容器内 `/opt/hyhal/bin/hy-smi` 直接可用。
- **dcu-ontrack-experience**：遇到 `libxxx.so` 缺失、编译失败、numpy/transformers 冲突等，去该 skill 按现象反查修复。
