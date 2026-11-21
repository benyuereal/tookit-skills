---
name: dcu-driver
description: "海光DCU驱动安装、版本兼容、加载验证与排障"
version: 1.2.0
author: DCU Engineer
license: MIT
---

# dcu-driver — 海光 DCU 驱动安装与验证

> 📌 **遇到具体报错先查排查经验库**：驱动加载失败、容器看不到卡、vbios/内核冲突等具体现象的处理经验（含 env/修复命令），统一收在 **`dcu-ontrack-experience`** 这个 skill 里。本 skill 只讲安装流程与兼容矩阵，具体操作手法请去那里按"现象关键词"反查，避免重复维护。

## 1. 驱动兼容列表（硬件 × 驱动 × DTK）

| 驱动版本 | 支持硬件 | 推荐 DTK 版本 |
|---|---|---|
| rock-4.0-23-5.6.25 | Z100 / Z100L | 21.04 / 21.10 / 22.04 |
| rock-4.5.2-5.11.40 | Z100 / Z100L | 22.04 / 22.10 / 23.04 |
| rock-5.2.0-5.16.18 | Z100 / Z100L | 22.10 / 23.04 |
| rock-5.2.0-5.16.29 | Z100 / Z100L / K100 | 23.04 / 23.10 |
| rock-5.7.1-6.2.26+ | Z100 / Z100L / K100 / K100-AI | 24.04 / 25.04 |
| rock-6.3.8+ | BW1000 / K100-AI / K100 / Z100L / Z100 | 25.04* |
| rock-6.3.30+ | BW1100 / BW1000 / BW150 / BW100 / BW10 / K100-AI / K100 / Z100L / Z100 | 26.04 |

> 选型原则：先按**硬件型号**锁定驱动大版本，再按目标 **DTK/OS** 版本选具体驱动。新硬件（BW 系列、K100-AI）只能用 6.3.x 以上驱动。

## 2. 驱动安装流程

### 2.1 安装依赖包

**CentOS：**
```bash
yum install -y rpm-build gcc-c++ cmake automake elfutils-libelf-devel libdrm libdrm-devel pciutils
yum install -y kernel-devel-`uname -r` kernel-modules-extra
```

**Ubuntu：**
```bash
apt install -y gcc g++ cmake automake libelf-dev libdrm-amdgpu1 libtinfo5 pciutils libdrm-dev
apt install -y linux-headers-`uname -r` linux-modules-extra
```

> ⚠️ **kernel 版本必须与 kernel-devel / linux-headers 版本一致**（`uname -r` 动态取值已处理）。
> ⚠️ **新版本驱动依赖 cmake > 3.2、gcc > 7.3**。安装前确认：`cmake --version`、`gcc --version`。

### 2.2 安装驱动

1. **下载驱动**：`<内部驱动镜像域>/6/main/latest驱动`（内部镜像域，按你环境填）
2. **赋权**：`chmod +x rock*.run`
3. **安装**：
   ```bash
   ./rock*.run          # 非虚拟场景推荐添加 -A 参数
   ```
4. **重启驱动服务**：
   ```bash
   systemctl restart hymgr
   ```

**安装注意事项：**
- 安装参数可用 `./rock*.run -h` 查看
- **安装新驱动无需先卸载旧驱动**：安装过程会自动卸载老驱动
- 安装中会遇 `Do you want to remove legacy hymgr config file? [Y/n]:`
  - **Y**：重新生成 `/etc/hymgr.cfg`
  - **N**：保留该文件（已有调优配置时选 N）
- 非 Z100L 环境会提示 `Update vbios? [y/N]:`
  - **首次安装建议 Y**（更新 vbios 后重启物理机）
  - 后续升级驱动可选 N
  - ⚠️ **云计算场景**：物理节点重启受严格管控，可不更新 vbios（Y 需重启物理机，N 不重启）

### 2.3 验证驱动加载

**6.2.x 及以下的驱动（模块名 `hydcu`）：**
```bash
lsmod | grep -E "hydcu|hycu"
# 预期：hydcu / hydcu_sched / hyttm / hykcl / hy_extra / amd_iommu_v2 / drm_kms_helper / drm
```

**6.3.x 及以上的驱动（模块名 `hycu`）：**
```bash
lsmod | grep -E "hydcu|hycu"
# 预期：hycu / hycu_sched / hyttm / hykcl / hydrm_buddy / hy_extra / iommu_v2 / drm_kms_helper / drm
```

> 🔑 **关键区分**：6.2.* 以前驱动模块名是 `hydcu`；**6.3.* 以后改为 `hycu`**。
> 验证脚本必须同时匹配两者（`grep -E "hydcu|hycu"`），否则 6.3 机器会误判"驱动未加载"。

## 3. /opt/hyhal 驱动安装目录与配套工具（<device-A> 实测，驱动 6.3.30）

> 来源：2026-08-28 在 <device-A> 设备（BW100X8）对 `/opt/hyhal` 全量扫描 + `--help`/安全只读命令实测。
> 装驱动后 `/opt/hyhal` 是**驱动运行时根目录**（容器必须 `-v /opt/hyhal:/opt/hyhal:ro` 挂载，见 dcu-ontrack-experience A 节）。
> ⚠️ 部分机器同时存在 `/usr/local/hyhal`（<device-A> 上是独立目录、bin 清单与 /opt/hyhal 完全一致，服务进程实际跑在 `/usr/local/hyhal/bin/` 下）——**两个路径的工具是同一套**，排障时两边都看一眼。

### 3.1 目录结构

```
/opt/hyhal
├── bin/          # 全部驱动配套工具（见 3.2）
├── lib/          # 驱动运行时库：libhsa-runtime64 / libamd_smi / librocm_smi64 /
│                 #   libhydmi / libhydrm / libhfmlib / librocm-dbgapi / lib/criu / lib/rocprofiler
├── include/      # amd_smi / amd-dbgapi / dmi 等头文件
├── hsa/          # hsa include + lib
├── dkms/         # 内核模块 .ko：hycu / hycu-sched / hyttm / hykcl / hydrm_buddy /
│                 #   hydrm_ttm_helper / hy-extra / hsw（DKMS 编译产物）
├── firmware/     # 0/（HCU 固件） driver/ hsw_firmware/ vbios/
├── asic_bin/     # asic_init_stage0~36.<ver>.bin + PspBootLoaderStage1 + readme.txt
│                 #   （partial vbios 板卡上电后需 push 的 ASIC 初始化固件）
├── bitcode/      # hyptxas-device-gfx906/926/928.bc（PTX 汇编器 device 库）
├── share/        # amd_smi / smi / gdb / man / modulefiles / rocprofiler-register / samples
├── miopen_cache/ # MIOpen kernel db 缓存（gfx*.HIP.*.ufdb.txt）
└── env.sh        # 驱动环境：LD_LIBRARY_PATH += lib lib/criu lib/rocprofiler；PATH += bin rocprofiler/bin libexec/rocprofiler
```

`source /opt/hyhal/env.sh` 后 `hfmtool`/`hdms`/`hymgr` 等才能找到 `libhfmlib.so`/`libhydmi.so`/`libhydrm.so`（**不 source 直接跑会报 `error while loading shared libraries`**，<device-A> 实测）。

### 3.2 bin 工具速查（用途 + 安全级别）

| 工具 | 用途 | 安全级别 |
|------|------|------|
| `hy-smi` | SMI 主工具（详见 dcu-hy-smi skill） | 只读项安全；`--hcureset`/`--unloaddriver` 等高危 |
| `hymgr` | 驱动管理守护（`systemctl restart hymgr` 管的就是它；<device-A> 上常驻 `/usr/local/hyhal/bin/hymgr`） | 服务进程，勿手动 kill |
| `hfm` | HCU 设备/计算管理服务（<device-A> 上常驻；`hfm -v` 报 `The hfm instance is running` 说明已有实例） | 服务进程 |
| `hdms` | HCU 设备管理服务（含虚拟化 `-p <proc> -v <vdev>`） | 服务进程 |
| `hfmtool` | **HCU/HySwitch 管理 CLI**：`-t` 取 HCU-HSW 拓扑、`-b` 取 PCI bus id、`--get_uuid`、`--get_nodes`（HSW id 列表）、`-l/-u` 加载/卸载 HCU 驱动、`-L/-U` 加载/卸载 HSW 驱动、`-c/-p/-P` HSW 配置、`-T/-f/-i` 刷固件（uboot/firmware/package，HCU id 0~7、HSW id 8~14）、`--reset_hys/--reset_all_hys/--reset_sys` 复位、`-r/-w/-v` 读写寄存器、`--open_dmsg` 开驱动日志 | ⚠️ 查询类（-t/-b/--get_*）只读；**刷固件/复位/卸载驱动/写寄存器高危**，仅产线/维修场景 |
| `hlinkmanager` | **HSL 卡间链路管理**：`-T` 列拓扑模板（PCIE_2HCU…OAM_7HSW_8HCU 等 12 种）、`-t <topo>` 设拓扑、`-r <128/160/256/320>` 设链路速率 | ⚠️ `-T` 只读（<device-A> 实测 `Detected 8 HCUs and 0 HSWs` + 模板表）；`-t/-r` 改链路配置高危 |
| `drvdiag` | **驱动诊断**：`-s` 扫 dmesg 出问题+处理指引（<device-A> 实测正常跑完 `Parseing 100% done`）、`-c` 抓 bug report 基础日志 | ✅ 只读，排障首选之一（规则引擎见 dcu-ontrack-experience F 节） |
| `mmio_rw` | **寄存器读写**：`-l` 列设备（<device-A> 实测 8 卡 BDF <device-BDF>~<device-BDF>）、`-s <bdf>` 走 PCI BAR、`-i <N>` 走驱动、`-r <reg>` 读、`-w <reg> <val>` 写、`dump/reset/driver` 子命令 | ⚠️ `-l` 只读；`-w`/`reset` 高危 |
| `dcufwloader_x86/_sw/_arm` | 固件加载器（v1.7.2，无 --help，参数见 `dcufwloader_x86` 裸跑输出） | ⚠️ 刷固件用，高危 |
| `run_pushAsic_and_HslLink.sh` | **partial vbios 板卡上电流程**：push asic_bin + 建 HSL link，交互选卡（单卡/多卡/全卡），状态机：full vbios 等 HSL → partial vbios 等 push asicbin → VBIOS inited 等加载驱动 | ⚠️ 维修/上电场景，生产机勿乱跑 |
| `update_ids` | 更新 PCI id 数据库（vendor 1d94 各 device id → 卡名映射：Z100/K100/BW1000/BW1100…） | 维护脚本 |
| `hiplogdump` | **HIP 日志解码**：`sort <log>` 按时间排序解压、`thread <log>` 按线程分文件、`unsort <log>` 不排序、`rcdf <log>` 调用间隔 RCDF | ✅ 只读，HIP_LOG 排障用 |
| `hycapture` | **HCU 调试器抓取**：`-p <pid>` 指定进程、`-a` 抓全部进程信息、`-nd` driverless 模式 | ⚠️ 会 attach 进程，先确认目标 |
| `hycoredump` | **core 文件解析**：`-i <core>` 输入（默认 core_0）、`-o <dir>` 输出、`-m` mem 目录、`-r` rocm 目录（默认 /opt/rocm）、`--dis_all` 反汇编全部 code object | ✅ 只读（见 dcu-vmfault-coredump） |
| `hyckptd` | checkpoint 守护（`--log <path>`、`-q`）；也可 `hy-smi --runhyckptd` 拉起 | 服务进程 |
| `hyptxas` | **PTX 汇编器**（v1.0.5）：`-offload-arch <gfx>`、`-o`、`-c`、`-compile-llvmir/-emit-llvmir`、`-clang-path/-rocm-path/-device-lib-path` | 编译工具 |
| `hycc` | HIP C 编译器前端（需 `HIP_CLANG_PATH` 指向 clang，裸跑报 missing clang path） | 编译工具 |
| `hyfatbinary` | fat binary 生成（`-create <file>`、`-image3`、`-offload-arch`、`-link` 等） | 编译工具 |
| `hybandwidth` | 带宽测试（需 HIP 运行时，裸机无 DTK 环境跑会 `Failed to load HIP library`——**容器内 source DTK 后跑**） | 测试工具 |
| `criu` / `criu-ns` | 进程 checkpoint/restore（依赖 `lib/criu` 的 libprotobuf-c 等，env.sh 已加 LD_LIBRARY_PATH） | 运维工具 |
| `event_rpcgen.py` | libevent rpc 代码生成（shebang 是 `python`，系统只有 python3 时跑不了，仅构建期用） | 构建工具 |

### 3.3 排障常用组合（<device-A> 实测）

```bash
source /opt/hyhal/env.sh
/opt/hyhal/bin/drvdiag -s                          # dmesg 驱动健康扫描（只读）
/opt/hyhal/bin/mmio_rw -l                           # 列 8 卡 BDF（只读）
/opt/hyhal/bin/hlinkmanager -T                      # HSL 拓扑模板 + 检测到的 HCU/HSW 数（只读）
/opt/hyhal/bin/hfmtool -t                           # HCU-HSW 拓扑（无 HSW 机型报 -805 属正常）
/opt/hyhal/bin/hy-smi --showdriverversion           # 驱动版本
/opt/hyhal/bin/hy-smi --showfwinfo                  # 各 HCU 固件版本
```

> ⚠️ `hfmtool -t/-b/--get_uuid` 在无 HSW 的 PCIe 机型（<device-A>）上返回 `state is -805/-810` 是**正常现象**（无 HSW 可查），不是故障；OAM+HSW 机型（BW1000/BW1100）才有真实拓扑数据。

### 3.4 手动刷 VBIOS（hyflash / vbios_upgrade.sh —— 官方部署 wizard，2026-10-14 摘录，未实测）

- **路径**：驱动装完后 VBIOS 固件与刷新工具在 `/opt/hyhal/firmware/vbios/`（**6.2.35-V1.6.7d 之前的驱动在旧路径 /opt/hyhal/vbios/**）。目录含 `hyflash`（及 `hyflash_x86/_arm/_sw` 变体）+ `vbios_upgrade.sh` + 按机型分的固件目录。
- **机型目录映射**：`BW3000/` = BW1000/BW150/BW100/BW10 系（如 `BW3000-6.312.002400Q.998006.bin`，另有 `-SRIOV-` 变体）；`NMZ/` = BW1100（如 `nmz_vbios_cl990865_gen4_256G_umc600M.bin`）；`KM/` = K100（`ai_k100*` PCIE 标卡、`hpc-*` 超算液冷）；`ZD/` = K100-AI（如 `K100_AI-6.314.001200r.889003.bin`）；`ZF/` = Z100L。
- **单卡刷新**：`cd /opt/hyhal/firmware/vbios && ./hyflash --index a --update ./BW3000/BW3000-<版本>.bin --xhcl`（K100-AI 例：`./hyflash --index a --update ./ZD/K100_AI-<版本>.bin`）。
- **一键自动**：`cd /opt/hyhal/firmware && ./vbios_upgrade.sh`。
- **与 .run 安装参数的关系**：`rock*.run -u 0/1/2` = VBIOS 自动/询问/跳过，`-A` 推荐（自动升驱动+配置+DCU VBIOS+HY Switch 固件），`-D` 只升驱动不动 VBIOS，`-H` hsw 固件模式；默认交互提示 `Update firmware of hsw? [Y/n]` 与 `Update vbios? [Y/n]`（见 §4 第 6 条）。
- **注意**：刷新前确认 VBIOS 与硬件版本匹配；带 VBIOS 升级后必须重启物理机（§4 第 6/9 条）；`lspci` 显 `rev ff`（卡掉落）时的恢复纪律见 dcu-ontrack-experience。

## 4. Pitfalls（实测/官方确认）

1. **驱动模块名 6.2/6.3 差异**：`hydcu`（≤6.2）vs `hycu`（≥6.3）。验证务必 `lsmod | grep -E "hydcu|hycu"`，单匹配一种会漏。本包验证机型均为 6.3.x（`hycu`）。
2. **容器内调试挂载**：**6.3.x 驱动下，docker 调用 DCU 时推荐挂载 `/sys/kernel/debug`**，否则容器内无法获取 DCU 进程信息（hy-smi `--showpids` 等失效）。docker run 加 `-v /sys/kernel/debug:/sys/kernel/debug` 或 compose 挂载。
3. **cmake/gcc 版本门槛**：新驱动要求 cmake > 3.2、gcc > 7.3。老旧 OS（如 CentOS 7 默认 cmake 2.8）需先升级，否则驱动编译/安装失败。
4. **kernel-devel 版本必须一致**：`uname -r` 与 `kernel-devel-$(uname -r)` / `linux-headers-$(uname -r)` 必须同版本，否则驱动模块编译失败。升级 kernel 后必须重装对应 devel 包再装驱动。
5. **hymgr 服务**：安装后 `systemctl restart hymgr` 是必须步骤（不重启服务，hy-smi / 设备节点可能不可用）。
6. **vbios 更新与重启**：非 Z100L 首次安装选 Y 更新 vbios 后**必须重启物理机**才生效；云环境因重启管控可选 N（牺牲 vbios 更新换稳定性）。
7. **`-A` 参数**：非虚拟（裸金属）场景推荐 `./rock*.run -A`（具体含义见 `./rock*.run -h`，通常与虚拟化/直通相关）；虚拟化/云场景按 -h 说明决定是否加。
8. **驱动自带的 hy-smi**：装驱动后 `hy-smi` 位于 `/opt/hyhal/bin/hy-smi`（宿主机与容器内均有，不一定在 PATH）——见 dcu-hy-smi skill。
9. **重装驱动后必须重启物理机**：驱动升级（尤其带 vbios 更新，见注意事项第 4 条）后，必须重启设备驱动才完全生效；不重启直接跑 rccl-test 会因驱动状态不一致导致带宽异常或设备不可用。来自内部测试手册"基本环境准备"。
10. **BIOS 关闭 IOMMU/ACS（rccl-test 前置，仅 PCIe 卡适用）**：关闭 IOMMU/ACS 的要求**主要针对 PCIe 形态卡**（K100-AI、BW10、BW100、BW150 等）——这类卡走 PCIe P2P 拓扑，BIOS 须关闭 ACS 和 IOMMU（或内核 `iommu=off amd_iommu=off`），否则通信带宽不达标。**OAM 形态产品（BW1000、BW1100）基本无影响，无需为此改 BIOS/内核**。检测见 dcu-env `scripts/check_iommu_acs.sh`。
11. **重启后驱动可能未自动加载**：部分设备装驱动并重启物理机后，驱动未自动加载，`hy-smi` 报设备不可用；需手动执行 `hy-smi --loaddriver` 重新加载驱动，再 `hy-smi -a` 验证设备可见。
12. **`/opt/hyhal` 与 `/usr/local/hyhal` 双路径**：部分机器（<device-A> 实测）两个目录并存且 bin 清单一致，常驻服务（hfm/hymgr）实际跑在 `/usr/local/hyhal/bin/` 下。排障时确认服务真实路径（`ps aux | grep -E "hfm|hymgr"`），工具两边都能用，但**版本以实际运行路径为准**。
13. **hyhal 工具不 source env.sh 报缺库**：`hfmtool`/`hdms`/`hymgr`/`criu` 等依赖 `/opt/hyhal/lib` 下的 `libhfmlib.so`/`libhydmi.so`/`libhydrm.so`/`libprotobuf-c`，直接跑报 `error while loading shared libraries`。先 `source /opt/hyhal/env.sh`（或 `LD_LIBRARY_PATH=/opt/hyhal/lib`）再跑。`hy-smi`/`drvdiag`/`mmio_rw`/`hlinkmanager` 不受影响。
14. **hfmtool 查询在无 HSW 机型报 -805/-810 属正常**：`hfmtool -t/-b/--get_uuid` 在纯 PCIe 无 HySwitch 机型（如 BW100）返回 `state is -805/-810, data_size ERROR` 是"无 HSW 可查"，不是故障；勿据此判驱动异常。
15. **dtk26.04 驱动搭配 BW150 做带宽测试异常**：特定驱动版本与 BW150 组合下带宽测试结果异常，需切换前期驱动版本使用。BW150 带宽测试前确认驱动版本，异常时先换驱动版本排除变量。

## 5. 与包内其他 skill 的衔接

- **dcu-env**：安装前用 dcu-env 检测 OS / kernel / 已装驱动版本，确定该装哪个驱动
- **dcu-hy-smi**：装完驱动后第一件事——`hy-smi -a` 验证设备可见；`/opt/hyhal/bin/hy-smi --showtoponuma` 验拓扑
- **dcu-topology**：驱动加载正常后，`dcu-topology/scripts/collect_topology.sh` 才能拿到真实卡↔NUMA（依赖 hy-smi）
- **dcu-docker**：容器内用 DCU 必须挂 `/dev/kfd` `/dev/dri` 且 6.3.x 额外挂 `/sys/kernel/debug`

## References

- 驱动下载：`<内部驱动镜像域>/6/main/latest驱动`（内部镜像域，按你环境填）
- 关联：dcu-env（环境检测）、dcu-hy-smi（驱动自带SMI）、dcu-topology（拓扑采集）、dcu-docker（容器挂载）
