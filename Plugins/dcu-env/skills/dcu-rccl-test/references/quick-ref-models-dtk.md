# 速查表：产品型号部署参数 + DTK 版本部署差异

> 测试前选型速查。
> 2026-10-16 自 `env/dcu-rccl-test/SKILL.md` 下沉（SKILL.md 瘦身到 500 行内），内容原文照录。

## 速查表：产品型号 → 部署关键参数

> ⚠️ 以下 gfx 架构与编译 commit 来自内部手册（覆盖 Z100L/K100/K100-AI/BW1000/BW1100）。**BW100 手册未覆盖，gfx/编译法按 BW1000 同架构（gfx936）类推，2026-09-16 <device-alias> 实测验证通过。**
> **BW150 已实测（2026-07-22）：gfx=gfx936（与 BW1000 同架构），用 BW1000 的 make 法 + commit `5e838ad...` 在 dtk26.04 下编译通过、单节点8卡测试跑通。**

| 产品型号 | 芯片ID | gfx 架构 | 编译 commit | 编译方式 | 备注 |
|---|---|---|---|---|---|
| Z100L | 54b7 | gfx906 | 5e838ad...（同 BW1000法） | make | — |
| K100 | 54b7? | gfx926 | 5e838ad... | make | 芯片ID待核 |
| K100-AI | 6210 | gfx928 | 5e838ad... | make | — |
| **BW1000** | 6320 | **gfx936** | `5e838ad9df47079e0e586ed38049f4f579ea462d` | make | DTK-25.04/25.04.1 可用 |
| **BW1100** | 6430 | **gfx938** | `66e513c24ff42394f5a0c1781f5868da7e094dd1` | `./install.sh` | 用 install.sh 非 make |
| BW100 | 6370 | **gfx936（2026-09-16 <device-alias> 物理机 rocminfo 实测，与 BW1000/BW150 同架构）** | `5e838ad9df47079e0e586ed38049f4f579ea462d`（同 BW1000 法，dtk26.04 实测编译通过） | make | 手册未覆盖，gfx/编译法按 BW1000 同架构类推并实测验证 |
| **BW150** | 6320 | **gfx936（实测）** | `5e838ad9df47079e0e586ed38049f4f579ea462d`（同 BW1000，dtk26.04 实测通过） | make | 芯片ID与BW1000相同、同gfx936；**容器测试实测** |

> 芯片ID↔产品型号映射完整版见仓库根 `references/dcu-models.md`。

## 速查表：DTK 版本 → 部署差异

| DTK 版本 | 关键差异 / 必做项 |
|---|---|
| DTK-25.04 / 25.04.1 | BW1000 用 `make` 编译法；fp8 需老分支（此版本已支持）；双机 topo 文件一般无需改 |
| **DTK-25.04.2** | **双机测试前必做**：按 `ibstat\|grep mlx5_\|wc -l` 修正 `topo_mapping_default.xml` 的 HCA 数；all_reduce 基线 117→**128** GB/s |
| 驱动 >= 6.3.30 | 建议开 `xdp_size=15` + `loaddriver`（需先停容器、卡无占用） |
| 其他/未列版本 | 以海光对应版本 Release Note 为准；commit 与编译法参考最近邻版本 |

> 实测基线（BW1000 双机16卡）：all_reduce algbw/busbw = 62/117 (25.04) → 68/128 (25.04.2)；alltoall = 30/27 → 31/29。
