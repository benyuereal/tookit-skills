# dcu-csv-passthrough — 海光 DCU 机密计算 + DCU 直通加密/明文虚拟机

海光 DCU 机密计算（CSV/CSV2/CSV3）+ DCU PCIE 设备直通加密/明文虚拟机（VFIO + sev-guest policy 0x1/0x5/0x45，明文直通删 sev-guest 即可）部署与踩坑：

- 机密计算原理（CSV 三代/隔离显存/SM2-SM4 密钥协商/CPU↔DCU 传输加密/安全页表/安全 command queue/多卡直通）
- 明文 & 加密双路直通（§3.3 加密 / §3.3.1 明文最简命令）
- 环境部署（BIOS Bootloader 按 CPU Model / AnolisOS 8.10 / 内核参数 csv_mem_percentage·mem_encrypt·kvm-amd.sev / hygon-devkit / qemu 编译 / IOMMU group 检查 / fixup header + 开机自启 / 明文平台 bridge 限定 BW150·BW1000）
- 启动前准备（解绑 hycu → VFIO 绑定 / 单卡直通 / NFS 挂载 / 镜像扩容）
- 虚拟机内装 DCU 驱动（tee 版 `CSV_PLATFORM=on` / 明文原厂驱动，含 VM 默认无网络·`virsh domifaddr` 查 IP·`virt-copy-in` 离线传文件踩坑）
- 大模型推理踩坑（DS-7B nfs 模型精度异常 vs modelscope 正常 / 多卡 P2P 受阻 `NCCL_P2P_DISABLE=1` / 多卡需关 `mem_encrypt=on` / DS-32B 双卡直通）
- BW1000 8 卡 + 7HSL 模组直通（8 卡直通脚本 / csv3_guest_rsv_mb / DS-V4 SGLang 启动脚本）
- 性能影响结论（HYswitch 架构 HSL 310GB/s 不受虚拟化影响 / 虚拟机直通 PCIe 降 17% / 低并发降 14-18.6% 高并发 1.2-7.1% / 加密虚拟机高并发损失 <2%，数据表 4-8 完整收录）
- 子文档溯源（4 篇 内部文档，含内部作者《虚拟化直通速成手册》明文直通补强）

> 54 个文档空截图占位（`![image]()`）已剔除；性能数据表为来源文档实测，作架构选型参考，部署前以实测为准，不视为固定基线。

详见 `SKILL.md`。
