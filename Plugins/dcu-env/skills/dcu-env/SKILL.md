---
name: dcu-env
description: "Detect DCU server env (CPU/NUMA/DTK/driver) pre-tuning."
version: 1.1.1
author: DCU Engineer
license: MIT
---

# dcu-env — 海光 DCU 环境检测（OS / kernel / 驱动 / 拓扑）

## 0. 操作范围门（scope gate，执行前必读）

本 skill 的采集类操作（dcuprofiletools、hy-smi、lspci、内核参数**读取**）是只读的，可直接执行；同一台机器上的**变更类**操作默认禁止：

- **改配置**：BIOS（ACS/IOMMU）、内核启动参数、sysctl、驱动重载/升级——先报确切命令与影响面，获用户批准再动。
- **共享机**：只操作本会话自己启动的进程/文件；他人环境默认只读。
- 例外（须动他人资源才能完成任务）：逐次向用户报确切命令获批准后执行；长期成立的例外登记到下方白名单，其余情形仍按默认规则。

**例外白名单（经用户批准后逐条登记）**：暂无

## 概述

在跑任何 benchmark / 调优 / 验收之前,必须先摸清目标 DCU 服务器的环境。本 skill 复用 vendored 的 **dcuprofiletools**(DCU 官方环境检查工具集)来完成采集,而不是重写检测逻辑。

dcu-env 的职责是:**通过 SSH 把工具传到 DCU 服务器 → 远程运行 → 取回结构化结果 → 解析为标准格式**,供 dcu-topology / dcu-llm-benchmark 等下游 skill 直接消费。

## 适用场景

- 首次连接一台 DCU 服务器,需要建立环境基线
- benchmark / 调优前确认硬件拓扑、驱动、DTK 版本
- 验收测试前做硬件健康检查(PCIe 链路、BAR、驱动加载)
- 售后排查硬件/驱动/固件问题
- 任何需要"这台机器到底什么配置"的场景

Don't use for: 性能压测本身(那是 dcu-llm-benchmark);纯本地操作。

## ⚠️ 测试前置检查：IOMMU / ACS 是否关闭（rccl-test 强相关）

`scripts/check_iommu_acs.sh` 专门检查 BIOS/内核是否已关闭 **IOMMU** 与 **ACS**。

> ⚠️ **前提条件（产品形态相关，先判断再处理）**：关闭 IOMMU/ACS 的要求**主要针对 PCIe 形态的卡**（如 K100-AI、BW10、BW100、BW150 等）——这类卡走 PCIe P2P 拓扑，IOMMU/ACS 开启会把 DCU 卡隔离、限制直连，导致 allreduce/alltoall 带宽不达标。而 **OAM 形态产品（如 BW1000、BW1100）基本不受影响**，无需为此专门改 BIOS/内核参数。动手前先确认设备是 PCIe 卡还是 OAM 模组，避免对 OAM 机型做无意义的高危 BIOS 改动。

rccl-test 通信测试对 PCIe P2P 拓扑敏感，IOMMU/ACS 开启（仅 PCIe 卡相关）会把 DCU 卡隔离、限制直连，
导致 allreduce/alltoall 带宽不达标。文档明确要求"BIOS 下已关闭 acs 和 iommu 选项"。

**调用时机**：在 dcu-rccl-test 跑任何通测前，先跑此检查；两者未全关则提示用户先关再测。

```bash
# 本地/远程均可
bash scripts/check_iommu_acs.sh
# 退出码：0=均已关闭(可测)；1=未全关(应提示用户关闭后再测)
# 输出 JSON(stdout) + 人类可读提示(stderr)
```

判定逻辑：
- **IOMMU 关闭** = `/proc/cmdline` 含 `iommu=off`/`amd_iommu=off` 或无启用证据（`/sys/kernel/iommu_groups` 为空）
- **ACS 关闭** = `lspci -vvv` 查 DCU 设备（1d94:）的 `ACS:` 行（权威判定，PCIe 扩展能力偏移不固定，不可 setpci 硬编码）；无 `ACS:` 行或 `Disabled` 即视为关闭；内核含 `pcie_acs_override=` 也视为关闭
- 两者未全关 → 输出 ⚠️ 提示，建议 BIOS 关闭 ACS/IOMMU，或内核加 `iommu=off amd_iommu=off`

> 实测（多机型，2026-07-22）：IOMMU disabled、ACS 能力行缺失（海光默认不启用），基线合格。

## ACS / IOMMU 查询指令速查（通用场景）

> 不止 rccl-test 前置检查，凡是要确认设备 PCIe 隔离 / 直通 / IOMMU 分组状态时，都可直接用下面的一键指令现场核对 BIOS 配置是否生效。

### 查 ACS（Access Control Services，PCIe 隔离）

```bash
# ① 一键判读：所有设备里只要有 SrcValid+ 就是 ACS 开启，全 - 即关闭
lspci -vvv | grep -q "SrcValid+" && echo "ACS 已开启" || echo "ACS 已关闭"

# ② 看具体设备的 ACSCtl 行（设备自身无则查上游 PCI bridge）
lspci -vvv -s <BDF> | grep "ACSCtl"
lspci -vvv -s <bridge_BDF> | grep "ACSCtl"
```

**判读标准**（ACSCtl 行）：

| 输出 | 结论 |
|---|---|
| `ACSCtl: SrcValid+ TransBlk+ ReqRedir+ CmpltRedir+ UpstreamFwd+ EBlk+` | **ACS 已开启** |
| `ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EBlk-` | **ACS 已关闭** |
| 设备自身无 `ACSCtl` 行 | ACS 由上游 PCI bridge/switch 决定，查上游 |
| 内核 cmdline 含 `pcie_acs_override=` | ACS 被内核绕过，视为关闭 |

> **注意**：`SrcValid`（源验证）是 ACS 隔离是否生效的最关键位，`SrcValid+`=开启、`SrcValid-`=关闭。ACS 是 PCIe 扩展能力（Ext Cap ID 0x000D），偏移因设备而异，**勿用 setpci 硬编码寄存器偏移**（见常见问题 #8）。

### 查 IOMMU（是否开启、分组情况）

```bash
# ① 内核启动参数是否显式开/关（海光是 AMD 系，对应 amd_iommu）
cat /proc/cmdline | grep -E "iommu|intel_iommu|amd_iommu"

# ② 开机日志里 IOMMU 是否注册
dmesg | grep -iE "iommu|DMAR"

# ③ IOMMU 分组是否启用（最权威：目录非空=启用）
ls -d /sys/kernel/iommu_groups/*/ 2>/dev/null | wc -l

# ④ 列出每个设备所属的 iommu group（看是否被 ACS 拆成独立组）
for g in /sys/kernel/iommu_groups/*/devices/*; do
  echo "$g" | sed 's#/sys/kernel/iommu_groups/##; s#/devices/.*# → #'
done
```

**判读标准**：

| 检查项 | 结论 |
|---|---|
| cmdline 含 `iommu=off` / `amd_iommu=off` | IOMMU 关闭 |
| `/sys/kernel/iommu_groups` 为空 | IOMMU 未启用（关闭） |
| cmdline 含 `iommu=on` / `amd_iommu=on` | IOMMU 开启 |
| `/sys/kernel/iommu_groups` 非空 | IOMMU 开启 |

> ① 同 bridge 下多设备挤在同一个 iommu group → ACS 未生效（否则会被拆成独立组）。
> ② PCIe 形态卡（K100-AI/BW10/BW100 等）IOMMU/ACS 开启会限制 P2P 直连；OAM 形态（BW1000/BW1100）基本不受影响。

## 内置工具（vendored，未随包分发）

海光官方 dcuprofiletools 为第三方 vendor（被 `.gitignore` 的 `**/vendor_*/` 排除），**不随本包分发**；需要时从官方渠道获取后放到本 skill 的 `scripts/vendor_dcuprofiletools/`：

```
scripts/vendor_dcuprofiletools/
├── dcu_performance_analyzer.py   # Python版,输出JSON报告,首选
├── env_check/                    # bash脚本集,深层硬件/驱动/固件诊断
├── dist/dcu_analyzer            # PyInstaller打包的可执行文件(无需Python)
└── README.md / DEV_GUIDE.md / USER_MANUAL.md
```

### 两套能力

| 工具 | 用途 | 输出 |
|---|---|---|
| `dcu_performance_analyzer.py` | 模块化采集 system/pcie/driver/logs/hardware/performance | JSON + 文本报告(首选) |
| `env_check/dcu_env_check-main/` | 深层诊断:PCIe BAR、SME、驱动内核不匹配、掉卡、主板问题 | tar.gz 打包日志 |

## 执行流程（强制）

### Step 1: 加载 dcu-ssh,连接设备

按 dcu-ssh 规范读取 `data/devices.json` + 外部 `credentials.json`,建立 SSH 连接。
**前置**:`sshpass` 必须已安装(本地和远端都建议有)。

### Step 2: 部署工具到远端

```bash
# 用 rsync 或 scp 把整个 vendor 目录传上去
rsync -avz -e "ssh -o StrictHostKeyChecking=no" \
  scripts/vendor_dcuprofiletools/ \
  root@<HOST>:/tmp/dcuprofiletools/

# 确认 hy-smi 可用（驱动自带，位于 /opt/hyhal/bin/hy-smi，无需 source DTK）
ssh root@<HOST> "ls -l /opt/hyhal/bin/hy-smi || docker exec \$(docker ps -q | head -1) which hy-smi || echo HY_SMI_NOT_FOUND"
```

### Step 3: 远程运行采集

**Python 版（推荐,有JSON输出）**
```bash
ssh root@<HOST> "cd /tmp/dcuprofiletools && python3 dcu_performance_analyzer.py -o /tmp/dcu_env_out"
# 指定模块(更快):
ssh root@<HOST> "cd /tmp/dcuprofiletools && python3 dcu_performance_analyzer.py -c system pcie driver -o /tmp/dcu_env_out"
```

**预编译版（远端无Python时）**
```bash
ssh root@<HOST> "cd /tmp/dcuprofiletools/dist && chmod +x dcu_analyzer && ./dcu_analyzer -o /tmp/dcu_env_out"
```

**方式C — 深层硬件诊断(bash脚本,售后面)**
```bash
ssh root@<HOST> "cd /tmp/dcuprofiletools/env_check/dcu_env_check-main && bash system_check.sh -o /tmp/dcu_hw_out"
```

### Step 4: 取回结果

```bash
# 报告(JSON + 文本)
scp -o StrictHostKeyChecking=no root@<HOST>:/tmp/dcu_env_out/reports/* ./

# 如需完整日志包(含 dmesg、lspci -vvv 等)
ssh root@<HOST> "cd /tmp && tar czf dcu_env_out.tgz dcu_env_out"
scp -o StrictHostKeyChecking=no root@<HOST>:/tmp/dcu_env_out.tgz ./
```

### Step 5: 解析为标准格式(关键)

dcu_performance_analyzer.py 的 `CheckResult` 结构:
```json
{
  "module": "System Information | PCIe Devices | Driver Status | ...",
  "status": "PASS | FAIL | WARNING | INFO",
  "message": "...",
  "details": { ... },
  "timestamp": "ISO8601",
  "execution_time": 0.0
}
```

dcu-env 智能体应将结果**归一化为 dcu-toolkit 统一环境快照**,写入 `data/`(或返回给调用方):
```json
{
  "device": "<设备别名>",
  "timestamp": "ISO8601",
  "cpu": { "model": "...", "numa_topology": {...} },
  "memory": "...",
  "dcu": {
    "count": 8,
    "type": "BW100",
    "pcie_topology": "...",
    "bar_errors": 0,
    "cross_numa_count": 0,
    "link_downgrade": 0
  },
  "dtk_version": "...",
  "driver_status": "PASS/WARNING/FAIL",
  "health_summary": { "PASS": N, "FAIL": N, "WARNING": N }
}
```

### Step 6: 清理

```bash
ssh root@<HOST> "rm -rf /tmp/dcuprofiletools /tmp/dcu_env_out"
```

## DCU 型号 PCIe ID 映射 (来自工具源码,权威)

`dcu_performance_analyzer.py` 内置的硬件ID→型号映射,供 topology 识别参考:

| PCIe ID | 对应产品型号 |
|---|---|
| 1d94:6370 | **BW100**（芯片ID 6370，真机实测） |
| 1d94:6320 | **BW150 / BW1000**（共用6320；6320是芯片/板卡级ID非产品名） |
| 1d94:6430 | **BW1100**（HYGON DCU-3G，用户实测） |
| 54b7 | Z100 |
| 55b7 | Z100L |
| 62b7 | K100 |
| 6210 | K100-AI |
| 6211 | K100-AI-ECO |

### 产品型号 → 产品级 Device ID（十进制，与上方芯片级 ID 并存）

> ⚠️ **层级区分**：上方是 `lspci -nn` 的**芯片/板卡级设备ID**（十六进制，形如 `1d94:62b7`），
> 本表是**产品级 PCIe Device ID（十进制）**，两者层级不同、不可混用。
> - agent 解析 `lspci -nn` 时拿到芯片级十六进制 ID（按上方表映射到产品型号）；
> - 本表产品级十进制 ID 用于**产品识别/资产归类**（如厂商工具、BMC 上报的十进制 Device ID）。
> - 例：K100 芯片级 `1d94:62b7`（十六进制），产品级 Device ID `8275`（十进制），非同一数值。

| 产品型号 | 产品级 Device ID（十进制） | 备注 |
|---|---|---|
| K100 | 8275 | |
| K100_AI | 8277、8278 | 含两个子设备 ID |
| BW1000 | 6320 | 与 BW150 共用产品级 ID |
| BW150 | 6320 | 与 BW1000 共用产品级 ID |
| BW100 | 6370 | |
| BW10 | 6360 | |

> ⚠️ **型号识别要点（产品型号 vs 芯片ID）**:
> - 用户所列为**产品型号**，非芯片名（不存在 "BW3000" 产品）。`1d94:xxxx` 是 lspci 芯片/板卡ID。
> - 实测/确认：BW100→6370，BW150→6320，BW1000→6320（与BW150共用），BW1100→6430。
> - 工具的 `dcu_id_map` 未含 `6370`/`6430`，对 BW100/BW1100 报"0张DCU"——dcu-topology 须自行解析 `lspci -nn | grep -i 'co-processor'`，用本表映射到产品型号。
> - **工具 `downgrade` 字段可能误报**：实测某 BW150×8 机型全部 `downgrade:true`，但 `lspci -vvv` 原文显示 8 卡均为 LnkCap=LnkSta=32GT/s Width x16（Gen5 x16 满速），**实际无降速**。判定链路状态请以 `lspci -vvv` 的 LnkCap vs LnkSta 原文为准，勿轻信工具的 downgrade 标记。
> - BW10 的实测芯片ID 待补充。

> ✅ **BW1100 实机确认（2026-09-15, <device-alias>/<<internal-ip>>）**：`lspci -nn` 显示 8 卡 `1d94:6430`（HYGON DCU-3G），`rocminfo` 确认 **gfx938**（DCU-3G 新代际），8 HCU + 7 HSW（OAM+HSW 形态）。驱动 6.4.0-V1.7.0b，hy-smi 1.26.0。设备为 **OAM 形态**，`check_iommu_acs.sh` 实测 `all_disabled=true`（iommu/acs 均关，与"OAM 不受 IOMMU/ACS 影响"一致）。卡↔NUMA 横跨 6 节点、卡间全 HSW 全互联（见 dcu-topology / dcu-hy-smi 补充）。
> ⚠️ **注（BW 系列 DTK 版本）**：<device-alias> 宿主机 `/opt/dtk` 指向 **DTK-25.04.4**，但 **BW 系列（含 BW1100）必须用 DTK-26.04 及以上版本**，宿主 DTK-25.04 **不满足要求**。实际编译/推理/验证请进 **DTK-26.04 容器**（如 `pytorch:2.7.1-kylin10sp3-dtk26.04-py3.10` 镜像内 `/opt/dtk`=DTK-26.04-rc4），**勿用宿主 DTK-25.04**，也勿在起容器时挂宿主 `/opt/dtk` 覆盖镜像自带 DTK。

## dcu_env_check dcu.go 源码映射表（deviceNames，补充权威性）

> 来自 dcu_env_check 的 DCUCollector，`init()` 自注册。映射表在源码内，供采集降级识别参考。
> 注意：**此表与上方 dcu-env「型号映射」表存在冲突部分，以实测为准**。

| PCIe ID | 源码标注型号 |
|---|---|
| 54b7 | Z100 |
| 55b7 | Z100L |
| 62b7 | K100 |
| 6210 | K100-AI |
| 6211 | K100-AI-ECO |
| 6320 | BW（源码标注通用 BW 型号；dcu-env 实测为 BW150/BW1000 共用） |
| 6430 | BW1100（HYGON DCU-3G） |

> ⚠️ **冲突说明**：dcu.go 把 `6320` 标为通用 BW，但 dcu-env 实测确认 BW100=6370、BW150/BW1000=6320 共用、BW1100=6430。
> 型号识别以 dcu-env 实测表为准（上方「型号映射」表），勿把 dcu.go 的 `6320=通用 BW` 当产品型号。
> 该表主要用于**降级采集时按设备 ID 识别 DCU 存在**（dcu.go 报"0 张 DCU"的根因之一）。

## DCU 设备代际识别流程（dcu_env_check 源码，lspci -Dn 权威）

> 来自 dcu_env_check 的 DriverCollector，`init()` 自注册。不依赖 hy-smi，纯 lspci regex 识别。

两步 regex 匹配（lspci -Dn 输出，格式 `<device-BDF> 0380: 1d94:54b7`）：

**Step A — zifang 代（1d94:5xxx）**
```regexp
([0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9])\s+\S+\s+1d94:5[0-9a-f]{3}
```
匹配到 → devName="zifang"，BDF 列表进 devs。命中即 return（不继续 Step B）。

**Step B — kongming/beimeizhou 代（1d94:6xxx）**
```regexp
([0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9])\s+\S+\s+1d94:6[0-9a-f]{3}
```
匹配到 → 取 devID 末两位（subDev），映射：
| 末两位 | devName |
|---|---|
| 61 | kongming |
| 62 | kongming |
| 63 | beimeizhou |
| 64 | beimeizhou |
其余 | unknown_2nd_gen |

**用途**：识别 DCU 设备所属代际（驱动/固件适配判断），供 dcu-driver / dcu-ontrack-experience 参考。

**PCIe deep-dump**：对每个检测到的 BDF 执行 `lspci -s <bdf> -vvvv`，逐设备 dump 完整 PCIe 扩展能力，写入 driver/pcie_vvvv.log —— 链路状态权威判定源（LnkCap vs LnkSta）。

**日志拷贝**：DriverCollector 拷贝 `/var/log/hfm`、`/var/log/hymgr`、`/var/log/rock-kernel` 到 driver/，拷贝 syslog/messages/kern.log —— 供 driver_health 规则引擎做 dmesg 分析。

## 常见问题（Pitfalls）

1. **远端没 source DTK** — DTK 工具链（hipcc/rocblas/hipblaslt 等）找不到。远程命令前加 `source /opt/dtk/env.sh`。
   **例外：hy-smi 不需要 DTK** — 它随 DCU 驱动安装在 `/opt/hyhal/bin/hy-smi`（宿主机与容器内均有，PATH 可能不含），直接调绝对路径即可，勿依赖 `source /opt/dtk/env.sh`。检查命令：`ls -l /opt/hyhal/bin/hy-smi || docker exec <容器> which hy-smi`。
2. **远端无 python3** — 改用 `dist/dcu_analyzer` 可执行文件
3. **不取回 reports 只取 data** — reports/ 里才有聚合 JSON,data/ 是原始文本
4. **忘记清理 /tmp** — 每次跑完清掉,避免占满磁盘
5. **直接 copy 工具到持久目录** — 建议每次传 /tmp,保持 skill 包为唯一真相源
6. **BW 系列型号识别失败** — 工具的 PCIe ID 映射不全,遇到未知 ID 记录到 dcu-models.md
7. **`docker exec -c` 跳过 .bashrc 导致性能异常** — 在容器内跑推理/benchmark 时，若用 `docker exec bash -c '...'`（非交互非登录 shell），不会加载 `/root/.bashrc`，导致 rocblas/hipblaslt 未正确加载，性能掉 15-20%。测评必须用 `docker exec -d bash -ilc '...'`（见 dcu-docker 测评章节）。同理，工具自身若依赖 DTK 环境变量，也要确保 exec 进了登录 shell。
8. **⚠️ ACS 状态判定不能用 setpci 硬编码寄存器偏移** — `check_iommu_acs.sh` 初版用 `setpci -s <dev> 0x070F.w` 读 ACS 控制寄存器最低位，但 ACS 是 PCIe 扩展能力（Ext Cap ID 0x000D），**偏移因设备/内核而异，0x070F 不是固定地址**，硬编码会误判（恰好该地址低位为 0 时漏报 enabled）。修正版改用 `lspci -vvv` 查 DCU 设备（1d94:）的 `ACS:` 行（`Enabled`/`Disabled`）做权威判定；无该行为视为未启用（海光平台默认不编译 ACS）。涉及 PCIe 扩展能力的探测一律走 `lspci -vvv` 文本解析，不要 setpci 猜偏移。

## 验证清单

- [ ] SSH 连接按 dcu-ssh 规范建立(凭证在包外,权限600)
- [ ] 工具成功传到远端 /tmp/dcuprofiletools/
- [ ] 远端 DTK 环境已加载(hy-smi 可用)
- [ ] 采集完成,reports/ 下生成 JSON + 文本报告
- [ ] 结果归一化为统一环境快照(含 cpu/dcu/dtk/health)
- [ ] 结果已回传本地
- [ ] 远端 /tmp 已清理
## 官方文档中心指针（2026-10-14 实测）

- 机型基础环境部署（BW 系列/K100-AI/Z100L·K100）版本树 `d9126ccd-226f-11f1-b0c2-0242ac150003`（13 叶）。
- DCU 硬件环境部署手册/服务器适配列表 `678457f6-268f-11f1-b0c3-0242ac150003`（5 叶）。
- DTK 环境安装/常用命令与作业调度/HPC 应用 `5b77899d-ad7c-11f0-afac-2e5fc4af73a0`（21 叶）。
- 取全文：`GET /api-doc/document/TreeDocument?VersionUUID=<树uuid>` 列目录、`DetailDocument?UUID=<叶uuid>`（匿名，2026-10-14 实测）；总索引见包内 `../_dcu-toolkit-shared/references/sourcefind/doc-center-index.md`。
