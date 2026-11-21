# diagnosis 规则引擎速查（dcu_env_check 源码提炼）

> 来源：dcu_env_check 的 internal/diagnosis/rules.go，7 条内置关联诊断规则。
> 每条格式：**规则ID → 关联检查项 → 诊断结论 → 修复建议**。这是**多检查项关联**的正向引擎，
> 与 dcu-ontrack-experience SKILL.md 的「现象反查」、F 节「dmesg 正向规则」互补。

## 规则表

| ID | 名称 | 关联检查项 | 诊断结论 | 修复建议 | 级别 |
|---|---|---|---|---|---|
| DRV-001 | 驱动安装异常 | driver_load 失败 + kernel_match 失败 | DCU 驱动未正确安装，可能存在内核版本不匹配或驱动编译失败 | 1.确认当前内核版本与驱动编译目标一致 2.重新执行驱动安装脚本 3.检查 dkms 状态: dkms status | FAIL |
| HW-001 | PCIe 链路降级 | pcie_link 警告 | PCIe 链路存在降级或速率协商异常，可能导致带宽不足 | 1.重新插拔 DCU 设备 2.检查 BIOS PCIe 设置（禁用 ASPM）3.更换 PCIe 插槽或线缆 | WARN |
| SYS-001 | 系统兼容性问题 | board_compat 失败 + sme 警告 | 老旧主板（AS 系列）与 SME 同时存在，满负载可能出现掉卡 | 1.更换为 BH 系列或更新的主板 2.在 BIOS 中关闭 SME 设置 | FAIL |
| DRV-002 | 虚拟化配置冲突 | vfio_error 失败 + driver_load 失败 | VFIO 配置与 DCU 驱动冲突，设备可能已被虚拟机占用 | 1.检查 vfio-pci 是否绑定了 DCU 设备 2.解除 VFIO 绑定: echo 0000:xx:00.0 > /sys/bus/pci/drivers/vfio-pci/unbind 3.重新加载 DCU 驱动 | FAIL |
| HLT-001 | 硬件错误告警 | xid_sxid 警告 | 系统日志中发现 DCU 硬件错误码（XID/SXID），可能存在显存或硬件故障 | 1.查看 dmesg 详细错误日志 2.重启节点观察是否复现 3.若持续出现，联系硬件厂商报修 | WARN |
| DRV-003 | 驱动服务未启动 | driver_load 失败 + kernel_match 正常 | 驱动与内核版本匹配，但模块未加载，可能是驱动服务未启动 | 1.手动加载驱动: modprobe hydcu 2.检查 systemd 服务状态 3.查看 dmesg 中驱动加载失败的详细原因 | WARN |
| SYS-002 | 系统异常告警 | log_analyzer 警告 | 系统日志中发现异常关键字（fail/error/warn/exception），需进一步排查 | 1.查看 output/logs/ 目录下的原始日志 2.根据错误关键字定位问题组件 3.结合其他检查项综合判断 | WARN |

## 使用说明（与排查纪律一致）

1. 本表为源码工具提炼的**正向关联规则**，仅供参考。实际环境（驱动版本、DTK、机型批次、固件）差异大，
   同样现象可能根因不同。使用前先核对**设备型号 + 驱动/DTK 版本 + 报错原文**是否匹配。
2. **规则命中 ≠ 必然修复**：规则给出的是"检查项同时失败/警告 → 指向某方向"的关联结论，
   需结合原始报告（report.txt）里的各 CheckResult 原文、dmesg、日志综合判断。
3. 涉及高危操作（重启节点/重装驱动/刷 vbios/改 BIOS）时，先确认是否在产线/客户环境，
   确认有带外恢复通道，优先在测试机/空闲节点验证；本表建议不是操作手册。
4. 硬件损坏类（UE 错误、PCIe 断链不可恢复等）走换卡/报修，不在本表软件可避范围。

## 关联

- 各检查项的具体判定逻辑见 dcu_env_check 源码的 checker 包（driver_health / pcie_link / board_compat / sme / vfio_error / xid_sxid / log_analyzer）。
- 现象反查入口（按报错关键词 grep）在 dcu-ontrack-experience SKILL.md 的 A/B/C/D/E 节。
- XID/SXID 错误码正向速查见 references/xid_sxid_errors.md。