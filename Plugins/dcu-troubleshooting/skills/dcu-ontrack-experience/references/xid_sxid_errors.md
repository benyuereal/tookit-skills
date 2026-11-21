# XID / SXID 错误码速查手册（references/xid_sxid_errors.md）

> 来源：HYGON DCU XID/SXID 错误 **Rev. 2.2.2**（2026/05，按驱动 hygon6.3v1.6.0 修正 XID 列表；PDF 存 `_dcu-research/DCU_XID_v2.2.2.pdf`）。
> 修订史：1.0.0(2024/10 首发期) → 2.0.0(2025/05 增 XID 列表/定位工具) → 2.1.0(2025/10 格式) → 2.2.0(2025/12 增 SXID) → 2.2.1(2026/02 增 XID 检索实例) → **2.2.2(2026/05 按驱动 hygon6.3v1.6.0 修正 XID 列表)**。

## 一、XID 与 SXID 是什么

- **XID（Exception ID）**：HYGON DCU 驱动生成并记录于**内核事件日志**的错误报告，表明 DCU 发生一般性错误（硬件错误 / 驱动故障 / 应用程序问题）。XID 错误代码含义在不同驱动版本中保持一致。
- **SXID（HySwitch XID）**：HYGON **HySwitch** 互联芯片驱动通过类似机制上报的与 HySwitch 相关的错误。HySwitch 是连接多个 HSL LINK 的芯片，HSL LINK 是海光设计的 DCU 互联接口（DCU 到 DCU 直连，用于服务器多 DCU 扩展）。

## 二、XID 错误查看方法

Linux 下 XID 以 `Exception ID（XID）` 格式统一编码并记录在**内核日志**（dmesg / syslog）：

- CentOS：`/var/log/messages`
- Ubuntu：`/var/log/kern.log`
- 推荐直接用 `dmesg` 命令获取

XID 信息格式（表 3-1 定义）：
```
[datex] hycu xxxx:xxx:xx.x: hycu:(serial:%s)XID:%d, pasid:%d, pid:%d, tgid:%d, sub block: %d, info:%s
```
| 分段 | 含义 |
|------|------|
| `[datex]` | dmesg 时间戳 |
| `xxxx:xxx:xx.x` | PCIe BDF |
| `serial:%s` | DCU 序列号 |
| `XID:%d` | XID 错误标识符 |
| `pasid:%d` | 驱动分配 pasid |
| `pid:%d` | 线程 ID |
| `tgid:%d` | 线程组 ID（进程 ID） |
| `sub block:%d` | XID sub-id |
| `info:%s` | XID 具体错误信息（寄存器值或简述，格式不固定） |

**错误定位工具**（v2.2.2 第 3.2 节，三个）：`hy-smi`（驱动自带，监控每卡状态/RAS/异常）、`HyFieldDiag`（板卡故障诊断；硬件相关 XID 无法经自修复流程恢复时运行，复测仍异常→报修）、`drvdiag`（驱动日志获取，随驱动安装；规则引擎见 dcu-ontrack-experience F 节）。

## 三、XID 错误列表（表 4-1）

列含义：HW Error=硬件错误 / Driver Error=驱动错误 / User APP Error=用户程序错误 / FB Corruption=显存损坏 / Thermal Issue=过热问题（✓ 表示该项为该错误的可能原因）。

| XID | 故障 (Failure) | HW | Driver | UserAPP | FB | Thermal |
|-----|----------------|----|--------|---------|----|---------|
> **v2.2.2 相对 2.2.0 的表 4-1 变更**（2026-05 版，按驱动 hygon6.3v1.6.0 修正；PDF pypdf layout 逐列核对）：
> - **新增**：23/24 MP0 CE/UE、25/26 MP1 CE/UE（均仅 HW，取代原 29/30 MCA CE/UE）；49 Bad pages num reaches threshold（HW）；302 VR HOT（HW）；303 SMU NO RMA（HW）。
> - **移除**：29/30 MCA CE/UE、111/112 MEC warning/error、124 GFX CU Harvest event、130 High single-bit ECC error rate。
> - **改号**：120（ECC page retirement / row remapping recording event）→ 121（ECC page retirement recording failure，v2.2.2 表内该行未标列）。
> - **列标注修正**：1 UMC CE 加 FB；2 UMC UE Driver→FB；50 去 Driver（剩 HW+Thermal）；51 UserAPP→Thermal；76 QCM fence timeout 补 Driver；82/87/190 UserAPP→FB；301 UserAPP→Thermal。3~18、77、81、160、200/201/300 未变。

| XID | 故障 (Failure) | HW | Driver | UserAPP | FB | Thermal |
|-----|----------------|----|--------|---------|----|---------|
| 1 | UMC CE | ✓ | | | ✓ | |
| 2 | UMC UE | ✓ | | | ✓ | |
| 3 | GFX CE | ✓ | | | | |
| 4 | GFX UE | ✓ | | | | |
| 5 | SDMA CE | ✓ | | | | |
| 6 | SDMA UE | ✓ | | | | |
| 7 | MMHUB CE | ✓ | | | | |
| 8 | MMHUB UE | ✓ | | | | |
| 9 | ATHUB CE | ✓ | | | | |
| 10 | ATHUB UE | ✓ | | | | |
| 11 | NBIO CE | ✓ | | | | |
| 12 | NBIO UE | ✓ | | | | |
| 13 | HDP CE | ✓ | | | | |
| 14 | HDP UE | ✓ | | | | |
| 15 | HSL CE | ✓ | | | | |
| 16 | HSL UE | ✓ | | | | |
| 17 | DF CE | ✓ | | | | |
| 18 | DF UE | ✓ | | | | |
| 23 | MP0 CE | ✓ | | | | |
| 24 | MP0 UE | ✓ | | | | |
| 25 | MP1 CE | ✓ | | | | |
| 26 | MP1 UE | ✓ | | | | |
| 49 | Bad pages num reaches threshold | ✓ | | | | |
| 50 | High temperature | ✓ | | | | ✓ |
| 51 | DCU isolation | ✓ | ✓ | | ✓ | ✓ |
| 76 | QCM fence timeout | | ✓ | | | |
| 77 | HDP timeout | ✓ | ✓ | | | |
| 81 | VM fault | ✓ | ✓ | ✓ | ✓ | |
| 82 | ATHUB error | ✓ | ✓ | | ✓ | |
| 87 | Driver firmware error | ✓ | ✓ | | ✓ | |
| 121 | ECC page retirement recording failure | | | | | |
| 122 | Row remapping recording failure | ✓ | ✓ | | ✓ | |
| 160 | PCIE link lost | ✓ | ✓ | | | |
| 190 | DCU initialization failure | ✓ | ✓ | | ✓ | |
| 200 | VBIOS initial failure | ✓ | ✓ | | | |
| 201 | VBIOS read failure | ✓ | ✓ | | | |
| 300 | SMU initial failure | ✓ | ✓ | | | |
| 301 | SMU hotpot notice | ✓ | ✓ | | | ✓ |
| 302 | VR HOT | ✓ | | | | |
| 303 | SMU NO RMA | ✓ | | | | |

> 说明（v2.2.2 第 5 章 XID 常见错误；该章为 2.2.x 新增，含分类表 5-1 与 5-2 RAS UE / 5-3 Critical / 5-4 Fatal / 5-5 Non-Fatal / 5-6 Init 清单、第 6 章处理策略表 6-1/6-2 预警清单，按类查表即可）：XID 301（SMU hotpot notice）等属**预警（Non-Fatal，驱动完成预定恢复流程）**，不影响作业运行，建议保持监控、提前安排维护；原 2.2.0 的 XID 130（High single-bit ECC error rate）在 2.2.2 已移除。分类口径：硬件类（UMC/GFX/SDMA/MMHUB 等 CE/UE）、驱动类、用户程序类（VM fault 等）、显存损坏类、过热类。

## 四、XID 错误处理策略（第 6 章要点）

- 先通过 `dmesg` 拿到 XID 码与 `info` 字段（寄存器值/描述），结合上表定位可能原因。
- CE（Correctable Error，如 UMC CE、GFX CE）一般为可纠正错误，监控为主；UE（Uncorrectable，如 UMC UE、GFX UE）为不可恢复错误，通常涉及硬件，按硬件 RMA 流程处理（不在本仓库软件规避范围）。
- 驱动/固件类（如 VBIOS/SMU initial failure、Driver firmware error）：检查驱动与固件版本、必要时重装驱动/刷新固件。
- 复位/隔离类（XID 51 DCU isolation）：按隔离流程处理。

## 五、SXID 错误列表（v2.2.2 表 9-1，互联模块 HySwitch）

| SXID | 故障 (Failure) | HySwitch Error | HFM |
|------|----------------|----------------|-----|
| 200 | HySwitch Drv Probe Failed | ✓ | |
| 201 | HySwitch Read Reg Timeout | ✓ | |
| 202 | HySwitch Fw Update Result | ✓ | |
| 203 | HySwitch Fw Version Verify Result | ✓ | |
| 204 | HySwitch HSL Link Chain UE | ✓ | |
| 205 | HySwitch HSL Link Chain CE | ✓ | |
| 206 | HySwitch R5 ERR | ✓ | |
| 207 | HySwitch PCIe Link Lost | ✓ | |
| 251 | HySwitch Drv Load failed | ✓ | |
| 253 | HySwitch Num not Match | ✓ | |
| 255 | HySwitch Linking error | ✓ | |
| 257 | HySwitch reset error | ✓ | |
| 259 | Reset DCU error without DCU Driver | ✓ | |
| 261 | Reset system reach max time | ✓ | |
| 262 | Read DCU register failed | ✓ | |
| 263 | Read DCU/HySwitch num failed | ✓ | |

> v2.2.2 表 10-4（HFM HW SXID 错误列表）另含**新增两条**：**264 DCU num less than expect**（DCU 数量和要求不匹配 → 节点 DCU 驱动初始化失败、资源无法使用）、**265 Some DCU have no HSL**（某些 DCU 的 VBIOS 不支持 HSL 互联）。

> SXID 200–207 属 **HySwitch HW Error**（v2.2.2 表 10-2）；251/253/255/257/259/261/262/263 等属 HFM 任务软件报错（v2.2.2 第 10.3/10.4 节，处理策略在表 10-3/10-5）。

## 六、SXID 错误分类与处理（v2.2.2 第 10 章要点）

SXID 主要分两类（v2.2.2 表 10-1）：
- **HySwitch HW Error**：HySwitch 板卡硬件模块错误报警（如 200/201/203/204/205/206/207）。
- **HFM SW Error**：HFM 任务软件报错（HFM 在系统启动时同步运行，负责加载 HySwitch 驱动和建链，完成后才向 HYMGR 发消息、由 HYMGR 加载 DCU 驱动）。

**通用建议**：对所有 SXID 故障，都建议**重启系统**；若重启后继续出现相同 SXID，联系 HYGON 售后技术支持。

**HySwitch HW SXID 处理策略（v2.2.2 表 10-3，节选）**：
| SXID | 错误 | 关注等级 | 处理策略 |
|------|------|----------|----------|
| 200 | HySwitch Drv Probe Failed | 严重错误，无法恢复则报修 | 1. 检查驱动和固件版本是否正常安装；2. 重启节点确认故障是否复现；3. 复现则运行故障侦测套件定位，无法恢复报技服 |
| 201 | HySwitch Read Reg Timeout | 严重错误，无法恢复则报修 | 同上流程（检查驱动/固件→重启复现→侦测套件→报修） |
| 203 | HySwitch FW Version Verify Result | 不影响使用，无需关注 | 1. 升级节点驱动对应版本固件包；2. 升级失败可报技服 |
| 204 | HySwitch HSL Link Chain UE | 严重错误，无法恢复则报修 | 1. 放弃当前作业，重启节点观察；2. 复现则重新部署驱动及固件；3. 不可恢复则侦测套件定位、故障卡报技服 |
| 205 | HySwitch HSL Link Chain CE | 不影响使用，需关注 | 1. 重启节点观察；2. 复现则重新部署驱动及固件；3. 不可恢复则侦测套件定位、报技服 |
| 206 | HySwitch R5 ERR | 严重错误，无法恢复则报修 | 1. 放弃作业，重启节点观察；2. 复现则重新部署驱动及固件；3. 不可恢复则侦测、报技服 |
| 207 | HySwitch PCIe Link Lost | 严重错误，无法恢复则报修 | 1. 重启确认可否恢复；2. 带外管理升级固件尝试恢复；3. 无法恢复报技服 |

> 根因一般为板卡驱动安装异常或 HySwitch 芯片固件烧录异常，客户可先自行排查。严重硬件错误（UE/R5/PCIe Link Lost）需节点 DC 修复或报修。

## 七、使用须知（与排查纪律一致）

1. 本手册为官方错误码参考，**仅供参考，不保证完全修复**。实际环境（驱动版本、DTK、机型批次、固件）差异大，使用前先核对**设备型号 + 驱动/DTK 版本 + 报错原文**是否匹配。
2. 涉及重启节点 / 重装驱动 / 刷固件 / 带外操作等，先确认是否在产线/客户环境、有无带外恢复通道，优先在测试机/空闲节点验证。
3. 硬件损坏类（UE 错误、PCIe 断链不可恢复等）走换卡/报修，非软件可避。

## 八、关联

- 本文件是 **XID/SXID 错误码正向速查**（错误码 → 含义/原因/处理），与 SKILL.md 主文「按现象关键词反查根因」互补。
- 实操类：XID 查看用 `dmesg` + `hy-smi`（见 dcu-hy-smi）；SXID 涉及 HySwitch 互联，排查见 dcu-env / dcu-driver。
- 源文档：HYGON DCU XID/SXID 错误 Rev. 2.2.2（海光信息技术股份有限公司；2.2.1 增 XID 检索实例章，2.2.2 按驱动 hygon6.3v1.6.0 修正表 4-1）。PDF 存 `_dcu-research/DCU_XID_v2.2.2.pdf`。
