---
name: dcu-dtk
description: DCU DTK开发工具包下载、安装与激活；仅物理机需装，容器自带
version: 1.3.0
author: DCU Engineer
license: MIT
---

# dcu-dtk — 海光 DCU DTK 开发工具包安装与激活

DTK（DCU Toolkit）是海光 DCU 的开发工具包，提供 HIP 兼容编译器、运行时、数学库与
框架支持（PyTorch DCU 版 / vLLM / SGLang 等编译运行依赖）。**物理机部署 CUDA 系框架
前必先装并激活 DTK**；docker 容器内镜像已自带 DTK，无需配置。

## 1. 分发源

> ⚠️ 明确指定保留以下原文 URL（含内网分发域），agent 直接可用。

- 最新 DTK 下载根：`https://download.sourcefind.cn:65024/1/main/latest`
  - 打开后**按操作系统选择**对应 DTK 包；**若无完全对应系统，选内核版本最接近**的那个
  - （2026-10-09 实测：`latest` 指向 **DTK-26.10**，37 个 OS 子目录 + 顶层 `Document/` 含 Release Note 等 29 份手册）
- DTK 文档：`https://download.sourcefind.cn:65024/1/main/latest/Document`
- 版本配套关系表：`https://download.sourcefind.cn:65024/file/1/DTK与驱动版本配套关系表.md`
  - 该表列出 DTK 版本 ↔ 驱动版本 ↔ 系统/内核 的配套要求，**每次使用前必须联网复核**（见 Pitfall #1）

## 2. 安装流程（仅物理机）

```bash
# 2.1 从 latest 选系统对应的 DTK 包下载（无对应系统则选内核最接近）
#     以下为示意，实际文件名/路径以官网 latest 目录实时为准
cd /tmp
curl -L -O "https://download.sourcefind.cn:65024/1/main/latest/<系统对应DTK包>.tar.gz"

# 2.2 解压到 /opt
tar -xzf <系统对应DTK包>.tar.gz -C /opt
#     解压后 /opt 下出现如 dtk-<version> 目录（具体目录名以包内为准）

# 2.3 建软链 /opt/dtk -> 解压目录（统一入口，框架/编译按 /opt/dtk 找）
ln -sfn /opt/dtk-<version> /opt/dtk
```

## 3. 激活（仅物理机）

```bash
source /opt/dtk/env.sh
# 验证：
which hipcc          # DTK 提供的 HIP 编译器应在 PATH
echo $DTK_PATH       # 应输出 /opt/dtk
hy-smi -a            # 驱动/设备可见（依赖 dcu-driver 已装）
```

> 激活即把 `/opt/dtk/bin` 等加入 `PATH`、设 `DTK_PATH` 等环境变量；
> **每次新 shell 跑框架/编译前都要 `source /opt/dtk/env.sh`**（或写入 `~/.bashrc`）。

## 4. 适用范围（重要）

| 环境 | 是否需要装/激活 DTK | 说明 |
|---|---|---|
| 物理机（裸金属） | **必须** | 下载解压 + 软链 + source 激活，框架编译运行前置 |
| docker 容器 | **不需要** | 基础镜像已自带 DTK，直接 `source /opt/dtk/env.sh` 或已在镜像 ENV 内，**不要重复下载安装** |

- 容器内若要确认 DTK 是否已激活：`which hipcc` / `echo $DTK_PATH`；为空则镜像未带，需换带 DTK 的镜像（见 dcu-docker）。
- 物理机激活后若进容器，容器内是独立 ENV，仍需在容器内单独 `source /opt/dtk/env.sh`（前提是镜像内 `/opt/dtk` 已存在——通常基础镜像有）。

## 5. 版本配套关系（基线快照 + 联网复核）

> ⚠️ **本段为基线快照（取自官方配套关系表），非实时数据。每次使用前必须联网打开配套关系表核查最新版本**：
> `https://download.sourcefind.cn:65024/file/1/DTK与驱动版本配套关系表.md`
> 以联网查到的最新 DTK / 驱动配套为准。
> 注：表中 `K100_AI` 即本包其他处写的 `K100-AI`。
> 注：DTK-26.10 行取自官方 Release Note（2026-09-28），DTK-26.04.1 行取自在线配套表（2026-10-09 复核）；其余行为配套表基线快照。

| DTK 版本 | 配套发布的驱动版本 | 驱动版本要求 | 硬件类型 |
|---|---|---|---|
| DTK-26.10 | rock-6.4.2-V2.1.0.run | >=6.4.2-V2.1.0（Release Note §3；官方配套表 2026-08-25 版尚未收录此行） | 未列（Release Note 未标硬件类型） |
| DTK-26.04.1 | rock-6.4.0-V1.7.0.20260810.run | =6.4.0-V1.7.0 | NMZ, BW, K100_AI, K100, Z100L |
| DTK-26.04 | rock-6.3.30-V1.4.1a.run | >=6.3.30-V1.4.1a | BW, K100_AI, K100, Z100L |
| DTK-25.04.2 | rock-6.3.16-V1.1.0a.run | >=6.3.16-V1.1.0a | BW, K100_AI, K100, Z100L |
| DTK-25.04.1 | rock-6.3.6-V1.9.0c.run | >=6.3.6-V1.9.0c | BW, K100_AI, K100, Z100L |
| DTK-25.04 | rock-6.3.3-V1.8.0.run | >=5.7.1-V1.7.0 | BW, K100_AI, K100, Z100L |
| DTK-24.04.3 | rock-5.7.1-6.2.26-V1.5.aio.run | >=5.7.1-V1.0 且 <=5.7.1-V1.6.3 | K100_AI, K100, Z100L |
| DTK-24.04.2 | rock-5.7.1-6.2.18-V1.1.2.aio.run | >=5.7.1-V1.0 且 <=5.7.1-V1.6.3 | K100_AI, K100, Z100L |
| DTK-24.04.1 | rock-5.7.1-6.2.17-V1.1.1.aio.run | >=5.7.1-V1.0 且 <=5.7.1-V1.6.3 | K100_AI, K100, Z100L |
| DTK-24.04 | rock-5.7.1-6.2.13-V1.0.1a.aio.run | >=5.7.1-V1.0 且 <=5.7.1-V1.6.3 | K100_AI, K100, Z100L |
| DTK-23.10.1 | rock-5.2.0-5.16.18-V01.9.2.run | >=5.2.0-V10.9.2 / >=5.7.1-V1.0 且 <=5.7.1-V1.6.3 | K100, Z100L |
| DTK-23.10 | rock-5.2.0-5.16.18-V01.9.2.run | >=5.2.0-5.16.18 / >=5.7.1-V1.0 且 <=5.7.1-V1.6.3 | K100, Z100L |
| DTK-23.04.1 | rock-5.2.0-5.16.18-V01.9.2.run | >=5.2.0-5.16.18 / >=5.7.1-6.2.13a 且 <=5.7.1-6.2.18 | K100 |
| DTK-23.04.2（待核验） | rock-4.5.2-5.11.40-V01.8.1.run | >=4.5.2-5.11.38 / >=5.7.1-6.2.13a 且 <=5.7.1-6.2.18 | Z100L |
| DTK-23.04 | rock-4.5.2-5.11.39-V01.5.run | >=4.5.2-5.11.38 / >=5.7.1-6.2.13a 且 <=5.7.1-6.2.18 | Z100L |
| DTK-22.10.1 | rock-4.5.2-5.11.38-V01.4.2.1a.run | >=4.5.2-5.11.38 / >=5.7.1-6.2.13a 且 <=5.7.1-6.2.18 | Z100L |
| DTK-22.10 | rock-4.5.2-0-5.11.36-1-V01.3.2.run | >=4.5.2-5.11.36 / >=5.7.1-6.2.13a 且 <=5.7.1-6.2.18 | Z100L |
| DTK-22.04.2 | rock-kernel-refactory-rock-4.5.2-0-5.11.34-V01.3.run | 仅支持当前一个驱动版本 | Z100L |
| DTK-22.04.1 | rock-4.5.2-0-5.11.32-Linux.rpm | 仅支持当前一个驱动版本 | Z100L |
| DTK-22.04 | rock-4.5.2-0-5.11.32-Linux.rpm | 仅支持当前一个驱动版本 | Z100L |
| DTK-21.10.1 | rock-4.0-23-5.6.25-3.10.0-957.el7.x86_64.rpm | 仅支持当前一个驱动版本 | Z100L |

- 选 DTK 时必须先确认**已装驱动版本**满足配套要求（见 dcu-driver），版本不匹配会导致
  hipcc 编译出的程序运行异常或设备不可用。
- 驱动版本查法：`hy-smi -a` 或 `cat /opt/hyhal/version`（若驱动已装）。

## 5.5 DTK-26.10 发布摘要（2026-09-28，Release Note DTK-261000-001-0915-916）

> 来源：官方 Release Note PDF（`DTK-26.10/Document/DTK 26.10 Release Note.pdf`）+ 2026-10-09 全量解包核对（md5 校验通过）+ <device-alias>（BW1100 gfx938）真机 GEMM 实测。
> **现状（2026-10-14 实测订正，替代 10-09 快照）**：`release/dtk2610` 主发布通道仍未开（HTTP 500）；但 **customer/dtk2610 已建仓且叶子级有货**——顶层索引枚举漏计，叶子级 `/customer/dtk2610/simple/torch/` 实测 16 个 wheel（含 2.7.1+das.opt1.dtk2610、2.7.1+dtk2610.2610041021），另有 vllm 0.25.1+torch2110、triton 3.8.0、flash-attn 2.8.4 等，全仓 ≥27 包；**nightly/dtk2610 更活跃（33 包，已出 cp312/py3.12 构建）**：torch 2.8.0/2.9.0+dtk2610（261010）、vllm 0.28.1+dtk2610.torch2110（261012）、mooncake-transfer-engine-shca 0.3.13.post1（261014）；customer/dtk26101-rc1、customer/dtk2610-openeuler、customer/dtk2610-yybu 已建仓；das_release/das1.8.5（20 包）；nightly/dtk2704 已预留（0 包）→ **DTK-27.04 在途**。框架轮已开始向 26.10 通道铺开；查/装 26.10 配套轮走 customer/nightly 通道（方法见 dcu-das-hub-api），dtk2604 仍是 release 通道默认。

**主线**：ROCm 7.2.1 + CUDA 12.9 适配，BW 系列 DCU 卡深度优化（BLAS/BLASLt/DNN），国产 OS 适配（统信/龙蜥/OpenEuler/Debian/ScaleOS），通讯库版本升级，LLVM 编译器小型化，GCVM 前后端分离。

**新功能（按模块）**：
- 驱动：HSL Link 状态检查、HySwitch1.5 互联、互联拓扑自适应、VMM、HBM Page Repair
- HIP：ROCm 7.2.1 / CUDA 12.9、大 size(4M-1G) D2D 传输优化、vmm fabric 支持、hipGraph CG 捕获
- RCCL：2.27.3 升级（**包内实测 2.27.7**，ROCm 7.2.3.0）、SMFree、拓扑自适应（免手动指定）、VMM、小数据对称内存算法
- hipprof：`--realtime-trace <0=db/1=rt/2=rt-db>`、`--orderby-totalsize`、`--pmc [default|read|write|compute|wave|util|memory]`、pmc 集群模式
- DCC：LLVM 小型化、GCVM 前后端分离（独立中后端 comgr/clang/lld/device-libs/offload）
- BLAS：rocm-7.2.1、Auto-Tuning 更新、官方称 BW1000/1100 Fp32/Fp16 GEMM +40~60%（实测见下）
- hipblasLT：rocm-7.2.1、W8A8 +20%、支持 hipDNN 特殊融合算子、官方称 BW1000/1100 BF16 GEMM +150%（实测见下）
- MIOpen：hipDNN 接口对齐 cuDNN、conv2D/3D/depthwise/group 优化、大量融合模式
- HipFFT：FFTW 接口（fftw3.h→hipfftw.h，-lfftw3→-lhipfftw）
- SPARSE：spmv 新增确定性 flat 算法（HIPSPARSE_SPMV_CSR_ALG2）
- SOLVER：ROCm 7.2.1、新增 GESDD（分治 SVD）/LASR、BDSDC/SYTRS 去除外部 LAPACK 依赖

**已知问题（含规避）**：
- 编译器 float FMA 精度 → `#pragma STDC_FP_CONTRACT OFF`
- 多进程同卡多 stream 性能 → `HIP_VISIBLE_DEVICES`
- rocBLAS Fp32 splitK 大 k 精度 → `ROCBLAS_ATOMICS_MOD=0`
- rocFFT 无 FP16；K100-AI 无 fp64（降级 pass）
- 驱动偶发 hip 不加载 → `systemctl restart hymgr` 等 1 分钟
- RCCL NMZ 跨机小数据 → `NCCL_IGNORE_CPU_AFFINITY=1`

**废弃**：Code Object V2 不可用（需 V3/V4/V5）；GPUFORT 弃 gfortran9.3 改 Intel 2021.3.0；hipFFT 删 ROCFFT_RTC_CACHE_PATH 环境变量（二进制内字符串仍在，疑内部残留）。

**OS 支持**：37 系统（Kylin v10 SP3/v11、Ubuntu 20.04/22.04/24.04、CentOS/Rocky/Anolis/openEuler/BCLinux/Alinux/RedHat/SugonOS/TencentOS/Opencloud/Debian/ScaleOS/UOS/NFS 等，详见 Release Note 表 3.1）。

**⚠️ GEMM 性能实测（<device-alias> = BW1100 gfx938，26.04-rc4 vs 26.10，2026-10-09）**：
- rocblas 方阵（Fp32/Fp16 4096³/8192³）：**持平**（±3% 噪声），+40~60% 未复现
- rocblas LLM 形状：M=128 N=8192 Fp16 **+41%**（稳定）；M=128 N=4096 **单次跑间波动 2.5 倍**（5198 vs 1763）
- hipblasLt BF16：最佳 kernel（`--algo_method all`）26.04=270496 / 26.10=270443 **完全相同**；但 **26.10 默认 heuristic 稳定选坏 kernel（~106k，慢 2.5 倍）**，26.04 heuristic 选好 kernel（220k~268k）
- 结论：Release Note 性能数字在 BW1100 未复现；**26.10 算法选择（heuristic）有回退/不稳定**——GEMM 莫名慢 2.5 倍时优先查 algo_method（rocblas kernel DB 新增 `TensileLibrary_lazy_gfx938.dat` lazy 加载，疑根源）
- 驱动兼容：26.10 官方要求 rock-6.4.2-V2.1.0，实测 HIP runtime 在 6.4.0-V1.7.0 上可正常初始化（hipGetDeviceCount 探针）

## 6. Pitfalls（实测/官方确认）

1. **每次使用前联网核查配套关系**：DTK 与驱动强绑定，官网配套表会更新。skill 内版本号为
   基线快照，**不构成承诺**。agent 执行安装前**必须用 curl 拉取最新配套表**，解析后与
   本 skill 第 5 节快照比对，以联网拉到的为准：
   ```bash
   # 该直链可直接 curl 获取（无需浏览器渲染），返回 UTF-8 表格文本
   curl -sL "https://download.sourcefind.cn:65024/file/1/DTK与驱动版本配套关系表.md"
   # 解析：提取 DTK 版本列与驱动版本要求列，与本 skill 第5节比对；
   # 若远程出现本快照没有的新 DTK 版本或驱动要求变更，以远程为准并告知用户。
   ```
   - 若运行环境无法访问该域名（纯内网隔离/无外网），则以本 skill 第 5 节快照为基线，
     并在输出中明确提示「配套表未联网复核，版本以快照为准，请人工确认」。
2. **容器不重复装 DTK**：docker 基础镜像已带 DTK，物理机下载解压那套**不要进容器**；
   容器内只需 `source /opt/dtk/env.sh`（若镜像 ENV 未自动加载）。误在容器内跑 tar 解压
   到 /opt 会污染镜像且与环境预期冲突。
3. **软链统一入口**：务必 `ln -sfn /opt/dtk-<version> /opt/dtk`，框架/编译脚本按
   `/opt/dtk` 定位；直接写死版本目录会导致升级时多处改路径。
4. **激活非持久**：`source /opt/dtk/env.sh` 只对当前 shell 生效。编译/跑框架的每条命令
   前都要 source，或写入 `~/.bashrc` / 构建脚本开头。CI/远程执行（ssh 非交互 shell）尤其
   容易漏 source 导致 `hipcc: command not found`。
5. **选包按内核而非仅看系统名**：同一发行版不同内核，DTK 包可能按内核区分。latest 目录
   无完全对应系统时，**选内核版本号最接近**的包，避免 glibc/内核符号不匹配。

## 7. 与包内其他 skill 的衔接

- **dcu-driver**：DTK 依赖驱动，先按 dcu-driver 装好并验证（`hy-smi -a` 可见），再装 DTK。
- **dcu-docker**：容器内 DTK 由镜像提供；选带 DTK 的基础镜像，不要容器内重装。
- **dcu-env**：安装前用 dcu-env 检测 OS / kernel / 已装驱动版本，确定该下哪个 DTK 包。
- **dcu-hy-smi**：装完 DTK 后 `source /opt/dtk/env.sh` + `hy-smi -a` 联合验证环境就绪。

## 8. DTK 内部目录结构与工具速查

> 容器里 DTK 实际长什么样、每个工具脚本干嘛、源码摘要，已独立成 skill：
> **`dcu-dtk-layout`**（实测于 <container_vllm015> 容器，DTK-26.04-rc4）。
> 它含：顶层结构/du、env.sh 关键变量、`bin`/`hip/bin`/`dcc/bin` 全部脚本用途与源码要点、
> `gpurun` NUMA 亲核+CU 掩码详解、lib 库分组、cuda 兼容层、常用命令速查，
> 以及 11 个工具脚本的**原样源码存档**（见 `dcu-dtk-layout/references/dtk-scripts/`）。
> 需要查 DTK 内部地图/工具时，加载 `dcu-dtk-layout`。

## References
- 官方文档中心《DTK 使用手册》版本树 `4fd79e96-ad7c-11f0-afac-2e5fc4af73a0`（60 叶：DTK 软件栈简介/使用 C++ 编写 DCU 程序/Fortran/AI 应用）。
- 官方文档中心《AI 框架使用手册》版本树 `5f2cd48c-ad7c-11f0-afac-2e5fc4af73a0`（10 叶：MIGraphX 安装、TensorFlow/PyTorch 常见网络使用/调试优化/框架扩展、第三方库转码）——此前报告记"HPC"系章名误读，已按叶名改正。取全文：`GET /api-doc/document/TreeDocument?VersionUUID=<树uuid>` 列目录、`DetailDocument?UUID=<叶uuid>`（匿名，2026-10-14 实测）；总索引见包内 `../_dcu-toolkit-shared/references/sourcefind/doc-center-index.md`。

- DTK 下载：`https://download.sourcefind.cn:65024/1/main/latest`
- DTK 文档：`https://download.sourcefind.cn:65024/1/main/latest/Document`
- 配套关系表：`https://download.sourcefind.cn:65024/file/1/DTK与驱动版本配套关系表.md`
- DTK-26.10 文档（含 Release Note，29 份手册）：`https://download.sourcefind.cn:65024/1/main/DTK-26.10/Document`
- **DTK 内部目录与工具全解（独立 skill）：`dcu-dtk-layout`**（容器内 DTK 实测速查 + 脚本源码库）
- 关联：dcu-driver（驱动）、dcu-docker（容器DTK）、dcu-env（环境检测）、dcu-hy-smi（验证）、dcu-cpu-affinity（NUMA 绑核，与 gpurun 呼应）、dcu-ontrack-experience（排障）

## 工具链踩坑（2026-10-20 P0 采集，四机实测）
- **物理机 rocminfo 必须用与驱动配套的新版 DTK**：DTK-25.04.2 在新驱动宿主（<device-C>）只枚举 CPU agent（DCU 不可见）；DTK-26.04 在 glibc 2.28 宿主需 conda libstdc++ 6.0.34 作 LD_LIBRARY_PATH shim（GLIBCXX_3.4.26）
- DTK-26.04 运行时库名 **libgalaxyhip.so.5**；缺它 rocminfo 报 127（"command not found" 假象，实为 loader 失败）→ LD_LIBRARY_PATH 含 `/opt/dtk*/hip/lib`（宿主）或 `/opt/dtk/lib`（多数镜像）
- Ubuntu 22.04 宿主 hipcc 编译报 `'cmath' file not found`（cuda_wrappers 的 #include_next 链）：`--gcc-toolchain=/usr` 实测无效；**`-isystem /usr/include/c++/11 -isystem /usr/include/x86_64-linux-gnu/c++/11` 有效**（经 hipcc 透传）
- **DCU `hipDeviceProp_t` 字段名与 ROCm 不同**（DTK-26.04）：v1 用 `multiProcessorCount`（非 multiprocessors）、`maxSharedMemoryPerMultiProcessor`（非 sharedMemPerMultiprocessor）、有 `l2CacheSize`/`gcnArchName`、**无 `regsPerThread`**；v2（`hipGetDeviceProp_t_v2`）另有 `uuid`/`regsPerMultiprocessor`/`maxBlocksPerMultiProcessor`/`reservedSharedMemPerBlock`/`clusterLaunch`
- `rocprof -i metric.txt` 自定义计数器 DTK-26.04 不可用（`Error: Context Create failed`，<device-A> 双卡 + <device-B> 容器 4/4 失败）→ 用 `hipprof --pmc --pmc-type 3`（默认集 ~100 计数器可用，含每 kernel arch_vgpr/sgpr）；详见 `dcu-perf-profiling` §PMC 计数器速查
