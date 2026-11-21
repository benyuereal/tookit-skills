---
name: dcu-csv-passthrough
description: "海光DCU机密计算CSV+PCIE设备直通虚拟机部署与踩坑"
version: 1.2.1
author: DCU Engineer
license: MIT
---

# 海光 DCU 机密计算 + DCU 直通加密虚拟机（dcu-csv-passthrough）

适用场景：在**海光机密计算（CSV/CSV2/CSV3）加密虚拟机**里把 **DCU 卡通过 PCIE 设备直通（VFIO + sev-guest）** 给虚拟机用，跑 AI 训练/推理并保护模型与数据安全。本文是「机密计算原理 + 环境部署 + 启动直通 + 虚拟机内装驱动 + 大模型推理踩坑 + BW1000 多卡模组直通 + 性能影响数据」的完整手册。

> **来源**：
> - 主手册《DCU安全：加密虚拟机直通速成手册》（纯自包含已全文抓取，39 个未解析 `[image]()` 截图占位——**已全部剔除**）
> - 性能分析《探究BW1000下虚拟机直通对DCU性能带来的影响》（15 个占位已剔除，**数据表 4-8 完整收录**）
> - 内部子文档（测试数据 xlsx & csv，file_id 已脱敏）：**CLI 不可达（read-file 失败），仅留溯源 URL，不编造数字**
> ⚠ **DCU 适配身份**：CSV 是海光国密机密计算；DCU 作为 PCIE 设备直通给加密虚拟机，CPU↔DCU 传输需 SM4 传输密钥加密（防 PCIE 总线嗅探）。测试环境 X7950 BW1000 / K100_AI。
> ⚠ **不存测试日志**：数据表为来源文档实测，作为架构选型参考；具体数字随硬件/驱动版本变化，部署前以实测为准。

## 0. 一句话机制

海光 CSV 加密虚拟机（sev-guest policy 选代）+ DCU 通过 VFIO 直通进虚拟机；DCU 驱动用 `dtk-*-tee`（CSV_PLATFORM=on）；大模型推理在虚拟机内 docker 跑 vLLM/SGLang TEE 镜像。

## 1. 机密计算原理（为什么需要传输加密）

- **CSV 三代**：CSV（内存加密）→ CSV2（加虚拟机状态加密）→ CSV3（CPU 内隔离，禁主机读写虚拟机内存，机密性+完整性）
- **隔离显存**：DCU 显存分普通/隔离显存，CSV 虚拟机用隔离显存存模型/用户数据，主机无法读写
- **密钥协商（SM2/SM4）**：DCU 安全处理器固件生成 SM2 公私钥 K1；DCU 软件栈随机生成 SM4 传输密钥 K2，用 K1_pub 加密发固件，后续 CPU↔DCU 传输用 K2 加密
- **CPU→DCU 传输**：CSV 加密内存数据 → 用 K2 重加密到共享内存 → SDMA 拷到共享显存 → CU 用 K2 解密到隔离显存
- **DCU→CPU 传输**：隔离显存 → CU 用 K2 加密到共享显存 → SDMA 到共享内存 → 软件栈用 K2 解密到 CSV 加密内存
- **安全页表 / 安全 command queue**：隔离显存的页表、command queue 本身存在隔离显存，由 DCU 安全处理器更新，防篡改
- **多卡直通**：软件栈与每张 DCU 协商 K2（多卡共用同一 K2）；卡间传输也用 K2 加解密

## 2. 环境部署

### 2.1 BIOS 要求
- Bootloader 版本按 CPU Model 确认：`grep -i model /proc/cpuinfo | grep -v "model name" | uniq`
  - model=4：Bootloader >= 3.5.3.1
  - model=6（74xx/54xx）：>= 3.7.3.37
  - model=6（34xx）：>= 3.7.1.115
- 机密计算 BIOS：RDSEED/RDRAND=Auto、SMEE=Enable、ASID Space Limit=1
- 安全固件设置 enable；DCU 直通要求 IOMMU + ACS 打开

### 2.2 操作系统
- AnolisOS-8.10（[镜像](https://mirrors.openanolis.cn/anolis/8.10/isos/GA/x86_64/AnolisOS-8.10-x86_64-dvd.iso)）
- 主机内核命令行（**关键**）：
```bash
sudo grubby --update-kernel=ALL --args="csv_mem_percentage=50 amd_iommu=on mem_encrypt=on kvm-amd.sev=1 kvm-amd.sev_es=1"
# 多卡测试可删 mem_encrypt=on（见 §5 踩坑）
grubby --update-kernel=DEFAULT --remove-args="mem_encrypt=on"
cat /proc/cmdline   # 重启后确认
```

### 2.3 hygon-devkit 安装
```bash
sudo chmod a+w /opt
cd /opt/ && git clone --depth 1 https://gitee.com/anolis/hygon-devkit.git && mv hygon-devkit hygon
cd /opt/hygon/common/install/ && ./install_hag.sh && cp /opt/hygon/bin/hag /usr/bin
yum install -y ca-certificates && hag general hgsc_import   # 导入通用安全证书
hag general version          # 查版本
hag csv platform_status      # 查固件版本
dmesg | grep CSV
```

### 2.4 公共依赖 + qemu 编译
```bash
sudo /opt/hygon/csv/install_csv_sw.sh   # 注释掉 qemu 行
cd /opt/hygon/csv/ && git clone -b master https://github.com/guanzhi/GmSSL.git
yum install -y gcc-c++ make cmake automake && sudo ./build_devkit.sh
# qemu 推荐 hygon-qemu
git clone https://gitee.com/anolis/hygon-qemu.git && cd hygon-qemu/ && sudo /opt/hygon/csv/build_qemu.sh
```

### 2.5 准备虚拟机 + IOMMU group 检查
```bash
wget -c https://download.sourcefind.cn:65024/directlink/8/c86-devkit/previous-release/csv/AnolisOS-8.10-x86_64-ANCK.qcow2 -O /opt/hygon/csv/vm.qcow2
# 查每个设备的 iommu group（DCU 需在独立 group，否则不能直通）
for d in $(find /sys/kernel/iommu_groups/ -type l | sort -n -k5 -t/); do
  n=${d#/iommu_groups/}; n=${n%%/*}; printf 'IOMMU Group %s ' "$n"; lspci -nns "${d##*/}"; done
# 若 DCU 不独立 group，需在 BIOS 开 ACS
```

### 2.6 Host 安装 fixup header 插件
```bash
wget https://download.sourcefind.cn:65024/directlink/8/c86-devkit/previous-release/csv/dcu/hycu_pci_fixup_header_0.4.4.bin
chmod +x hycu_pci_fixup_header_0.4.4.bin && sudo ./hycu_pci_fixup_header_0.4.4.bin
cat /sys/module/hycu_pci_fixup_header/version   # 查版本
lspci -nn | grep "Co-processor"                 # 查 Vendor/Device ID（1d94:6320）
```

#### 2.6.1 插件补丁详细流程（明文直通共用，开机自启）
> 来自《虚拟化直通速成手册》（原作者，file_id `<file_id>`）的补充，加密/明文直通都走同一套 fixup header。
```bash
# 1) 先卸载旧插件（若之前装过）
rmmod vfio-pci
rmmod hycu_pci_fixup_header
# 2) 安装补丁（交互式 .bin）
sh hycu_pci_fixup_header_0.4.8.bin      # 版本号按实际下载文件
# 3) 关联 modprobe 命令
depmod
# 4) 检查加载的补丁模块
lsmod | grep -i fix
#    hycu_pci_fixup_header    28672  0
# 5) 版本查看
modinfo hycu_pci_fixup_header | grep -i version
#    version:        0.4.4
# 6) 加载 vfio-pci 模块
modprobe vfio-pci
lsmod | grep -i vfio
#    vfio_pci / vfio_pci_core / vfio_virqfd / vfio_iommu_type1 / vfio / irqbypass
# 7) 开机自启（关键，否则重启后需重跑）
echo "vfio-pci" > /etc/modules-load.d/vfio.conf
```
正常加载后 `dmesg | grep fix` 应看到每个 DCU 设备：
`hycu-fixup-header <device-BDF>: add flags NO_BUS_RESET` / `enable pcie atomic`（NO_BUS_RESET + PCIe atomic 是直通能稳定跑的前提）。
设备号核对：`lspci | grep Co-` → `<device-BDF> Co-processor: Chengdu C-3000 IC Design Co., Ltd. BW (rev 01)` 等。

### 2.6.2 官方下载位与手册索引（2026-10-14 官方渠道实测，未真机复测）

- **fixup header 插件**：z-file Cat5「DCU直通插件」`hycu_pci_fixup_header v0.4.8`（较 §2.6 示例的 0.4.4 新增 **NMZ/BW100** 机型支持；下载位同 `previous-release/csv/dcu/`，文件名 `hycu_pci_fixup_header_0.4.8.bin`）。
- **直通手册 PDF**：同 Cat5 下 v1.0.0 / v2.0.0 两版官方手册。
- **csv-tee 驱动**：Cat6「csv-tee驱动」→ `dtk-26.04-tee驱动/rock-6.3.32-V1.6.0.run`（见 §6.4/§6.5）。
- **C86 安全标准文档**（Cat8 C86 Devkit → `standards/`）：CSV3 API 文档 V1.5、C86 安全计算架构规范 V1.6、金融行业海光芯片安全白皮书；`psp-secure-firmware/`（C86-3G/4G/4G-H + `version_rules.json`）。
- **文档中心《DCU直通用户手册》**（在线版，按机型分叶 8 篇）：版本树 `0a12cd95-1b62-11f1-a4a0-0242ac150003`，叶=简介/环境配置/插件安装部署/虚拟化环境部署/BW1000·BW1100/BW150/BW100·BW10/Z100L·K100·K100-AI；取全文 `GET https://developer.sourcefind.cn/api-doc/document/DetailDocument?UUID=<叶uuid>`（匿名，Type=md）。§6 各机型章节可与对应叶互校。

### 2.7 明文直通平台限定条件（非加密直通专属）
> 仅明文（非 CSV 加密）直通需注意的机型/桥接约束（来源同上）：
- **BW150 带桥接器**：4 卡直通照常安装驱动，可自动走 bridge 无需额外映射
- **BW150 2 卡直通**：不支持走 bridge，安装驱动需 `bash xxx.run -t NONE`
- **BW1000**：只支持将所有 bridge 映射进 **1 个**虚拟机，其余虚拟机只能走 PCIe（不能多个 VM 各自分 bridge）

> 加密（CSV）直通不受上述 bridge 规则约束，见 §6 BW1000 8 卡 + hyswitch 直通。

## 3. 启动前准备（每次开机重做）

### 3.1 解绑 hycu 驱动（物理机已装驱动时）
```bash
lspci -s <device-BDF> -v | grep "Kernel driver"   # 查当前绑定
# 若绑 hycu：echo <device-BDF> | sudo tee /sys/bus/pci/devices/<device-BDF>/driver/unbind
sudo systemctl disable hymgr.service hfm.service   # 永久禁 Host 自动加载 DCU 驱动（重启生效）
```

### 3.2 加载 VFIO + 绑定 DCU
```bash
sudo modprobe vfio_virqfd vfio_pci vfio_iommu_type1 vfio   # 每次重启后需重加载
echo 1d94 6320 | sudo tee /sys/bus/pci/drivers/vfio-pci/new_id
lspci -s <device-BDF> -v | grep "Kernel driver"; ls /dev/vfio
```

### 3.3 单卡直通启动命令（**关键**）
```bash
qemu-system-x86_64 -name csv-vm --enable-kvm -cpu host -smp 32 -m 64G -nographic \
  -hda /opt/hygon/csv/vm.qcow2 \
  -drive if=pflash,format=raw,unit=0,file=/opt/hygon/csv/OVMF_CODE.fd,readonly=on \
  -object sev-guest,id=sev0,policy=0x45,cbitpos=47,reduced-phys-bits=5 -machine memory-encryption=sev0 \
  -device vfio-pci,host=<device-BDF> \
  -fw_cfg name=opt/ovmf/X-PciMmio64Mb,string=131072 \
  -net user,hostfwd=tcp::2222-:22 -net nic
```
> **policy 取值**：`0x1`=CSV、`0x5`=CSV2、`0x45`=CSV3（启动普通非加密虚拟机则删除整行 sev-guest）。`-fw_cfg X-PciMmio64Mb`：64G VRAM 用 131072，32G 用 65536；多卡线性调整。多卡加 `-device vfio-pci,host=xx:yy.z` 并相应调 mmio。
> 启动后 `ssh -p 2222 root@0.0.0.0`（密码 root）；`dmesg | grep CSV`、`lspci | grep Co-processor` 验证直通成功。

#### 3.3.1 明文（非加密）直通最简命令
> 来源《虚拟化直通速成手册》：**不传 `-object sev-guest ... -machine memory-encryption=sev0`** 即普通 VM 把 DCU 透传进去，VM 内直接装原厂驱动即可（原生性能、完整驱动 API、兼容性广、安全配置透明）。
```bash
qemu-system-x86_64 -name plain-dcu-vm --enable-kvm -cpu host -smp 32 -m 64G -nographic \
  -hda /path/to/vm.qcow2 \
  -device vfio-pci,host=<device-BDF> \
  -net user,hostfwd=tcp::2222-:22 -net nic
# 多卡：叠加 -device vfio-pci,host=xx:yy.z；mmio/fw_cfg 同上按显存调整
```
> 图形化创建 VM（可选）：`virt-manager` → 创建虚拟机 → 添加硬件 → PCI 主机设备 → 按设备号（如 `<device-BDF>`）选 DCU。

### 3.4 参考：NFS 挂载物理机目录 / 镜像扩容
- NFS：服务器端 `/etc/exports` 加 `/data 192.168.122.0/24(rw,sync,no_root_squash,insecure)`（`insecure` 允许非特权端口），`exportfs -arv`
- 扩容：`qemu-img resize vm.qcow2 +500G`，虚拟机内 `growpart /dev/sda N && resize2fs /dev/sdaN`

## 4. 虚拟机内环境安装

### 4.1 安装 DCU 驱动（**tee 版**）
```bash
wget -c https://download.sourcefind.cn:65024/directlink/8/c86-devkit/previous-release/csv/dcu/dtk-25.04.1-tee/rock-6.3.14-V1.13.0-tee.run -O rock-6.3.14-V1.13.0-tee.run
chmod +x rock-6.3.14-V1.13.0-tee.run
yum install -y rpm-build gcc-c++ cmake automake elfutils-libelf-devel libdrm libdrm-devel pciutils kernel-devel-`uname -r` kernel-modules-extra
CSV_PLATFORM=on ./rock-6.3.14-V1.13.0-tee.run   # 关键：CSV_PLATFORM=on
# 装完禁虚拟机 selinux（/etc/selinux/config -> SELINUX=disabled）
```

#### 4.1.1 明文（非加密）直通：VM 内装原厂驱动
> 来源《虚拟化直通速成手册》。明文直通无需 `CSV_PLATFORM=on`，装普通驱动即可（BW150 2 卡场景见 §2.7 的 `-t NONE` 约束）。
```bash
# centos：
yum install -y rpm-build gcc-c++ cmake automake elfutils-libelf-devel libdrm libdrm-devel pciutils
yum install -y kernel-devel-`uname -r` kernel-modules-extra
# ubuntu：
apt install -y gcc g++ cmake automake libelf-dev libdrm_amdgpu1 libtinfo5 pciutils libdrm-dev
apt install -y linux-headers-`uname -r` linux-modules-extra
# 安装驱动（普通版，不带 -tee / 不带 CSV_PLATFORM）
bash xxx.run
systemctl restart hymgr
/opt/hyhal/bin/hy-smi        # 查卡，确认 VM 内 DCU 可见
```

### 4.2 部署 docker + TEE 镜像 + 简易用例
```bash
yum install -y docker
docker pull haigon.cn/test/vllm:0.8.5-ubuntu22.04-dtk25.04.1-py3.10-tee
docker run -it --name csv_test -u root --privileged --network=host --ipc=host \
  --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri -v /opt/hyhal:/opt/hyhal \
  --group-add video --cap-add=SYS_PTRACE --ulimit stack=-1:-1 --ulimit memlock=-1:-1 \
  --security-opt seccomp=unconfined -v /data:/data \
  haigon.cn/test/vllm:0.8.5-ubuntu22.04-dtk25.04.1-py3.10-tee /bin/bash
# 简易用例
git clone https://github.com/ROCm-Developer-Tools/HIP-Examples.git && cd HIP-Examples/vectorAdd && make
HIP_LOG_LEVEL=4 HIP_MODULE_MASK=0x7fffffff HIP_ENABLE_LIST=hip HIP_LOG_OUTPUT_PATH="./" ./vectoradd_hip.exe
```

### 4.3 虚拟机内网络 / 离线传文件踩坑（明文直通共用）
> 来源《虚拟化直通速成手册》。明文与加密直通 VM 都适用。
- **VM 默认无网络**：新建 VM 进系统后默认没开网络，需手动开启；`virsh domifaddr anolis8` 可查 VM IP（默认类似 `192.168.122.218`，**不是** `192.168.122.1`——那是宿主机网关）
- **SSH 进 VM 更方便**：`ifconfig` 拿到 VM IP 后，宿主机 `ssh root@<vm_ip>` 操作（进 VM 后 `lspci` 应能看到 DCU 设备）
- **离线传文件**：宿主机用 `virt-copy-in` 把文件塞进 VM 磁盘
  ```bash
  yum install -y libguestfs-tools
  virt-copy-in -d anolis8 /root/rock-6.3.28-V1.3.0b.run /root   # 把驱动 run 包传入 VM 的 /root
  ```
- 驱动下载：https://download.sourcefind.cn:65024/6/main （选普通版 `rock-*.run`，非 `-tee`）

## 5. 大模型推理踩坑经验（**高价值**）

- **DS-7B 精度异常**：通过 NFS 挂载的模型启动测试精度异常；在容器内用 modelscope 直接下载模型则测试正常（模型文件一致性问题，sha256sum 对比一致但 NFS 挂载下推理异常——优先容器内下载模型）
- **多卡 P2P 受阻（重要）**：安全特性开 ACS 会使虚拟机内 P2P 受阻 → `NCCL_P2P_DISABLE=1` 后可正常加载模型
- **多卡必须关 mem_encrypt=on（重要）**：多卡模型加载后 cuda_graph 阶段报 `DMA mapping failed`，调 iommu pt / ulimit / 换 14B 均无效；**根因是 Host 多卡用 PCIE P2P 时 iommu 页表带上加密控制位导致 DCU 无法工作**。解决：**多卡测试关闭 Host 的 `mem_encrypt=on`**。注意该参数只控物理机内存加密，**不影响加密虚拟机的内存加密**（虚拟机内存加密由 sev-guest 控制）
- **DS-32B 双卡直通**：`-device vfio-pci,host=<device-BDF> -device vfio-pci,host=<device-BDF>`，`-fw_cfg X-PciMmio64Mb,string=262144`（双卡翻倍）

## 6. BW1000 8卡 + 7HSL 模组直通

> 前期 DS-32B 加密虚拟机性能与宿主机差距过大，故尝试模组（HSL）直通。

### 6.1 前期环境（龙蜥）
- 主机 AnolisOS-8.10，内核 5.10-anolis（commit a33879e5cc7d）；qemu 8.2.2（devel-8.2，打 pr_97 patch）
- 虚拟机内核须与宿主机同为 5.10-anolis（否则启动失败）
- 虚拟机镜像 AnolisOS-8.10，edk2：`yum install -y edk2-ovmf`

### 6.2 VFIO 绑定（DCU + hyswitch）
```bash
echo 1d94 6320 | sudo tee /sys/bus/pci/drivers/vfio-pci/new_id   # DCU
echo 1d94 a110 | sudo tee /sys/bus/pci/drivers/vfio-pci/new_id   # hyswitch
lspci -s <device-BDF> -v | grep "Kernel driver"; lspci -s <device-BDF> -v | grep "Kernel driver"
ls /dev/vfio   # 产生 DCU/hyswitch 所在 iommu group 的 ID 文件
```

### 6.3 8 卡 + hyswitch 直通启动（CSV3）
```bash
sudo modprobe vfio_virqfd vfio_pci vfio_iommu_type1 vfio
lspci -nn | grep "Co-processor"; lspci -nn | grep "Bridge \[0680\]"
echo 1d94 6320 | sudo tee /sys/bus/pci/drivers/vfio-pci/new_id
echo 1d94 a110 | sudo tee /sys/bus/pci/drivers/vfio-pci/new_id
sudo qemu-system-x86_64 -name csv-vm --enable-kvm -cpu host -smp 128 -m 512G -nographic -machine q35 \
  -hda /opt/hygon/csv/vm.qcow2_new \
  -drive if=pflash,format=raw,unit=0,file=/opt/hygon/csv/OVMF_CODE.fd,readonly=on \
  -object sev-guest,policy=0x45,id=sev0,cbitpos=47,reduced-phys-bits=5 -machine memory-encryption=sev0 \
  -net user,hostfwd=tcp::2222-:22 -net nic \
  -device pcie-root-port,id=rp1,chassis=1,slot=1,bus=pcie.0,multifunction=on,addr=0x5.0 \
  -device pcie-root-port,id=rp2,chassis=1,slot=2,bus=pcie.0,addr=0x5.1 \
  -device pcie-root-port,id=rp3,chassis=1,slot=3,bus=pcie.0,addr=0x5.2 \
  -device pcie-root-port,id=rp4,chassis=1,slot=4,bus=pcie.0,addr=0x5.3 \
  -device pcie-root-port,id=rp5,chassis=1,slot=5,bus=pcie.0,addr=0x5.4 \
  -device pcie-root-port,id=rp6,chassis=1,slot=6,bus=pcie.0,addr=0x5.5 \
  -device pcie-root-port,id=rp7,chassis=1,slot=7,bus=pcie.0,addr=0x5.6 \
  -device pcie-root-port,id=rp8,chassis=1,slot=8,bus=pcie.0,addr=0x5.7 \
  -device vfio-pci,host=<device-BDF>,id=vfio_dcu1,bus=rp1 \
  -device vfio-pci,host=<device-BDF>,id=vfio_dcu2,bus=rp2 \
  -device vfio-pci,host=<device-BDF>,id=vfio_dcu3,bus=rp3 \
  -device vfio-pci,host=<device-BDF>,id=vfio_dcu4,bus=rp4 \
  -device vfio-pci,host=<device-BDF>,id=vfio_dcu5,bus=rp5 \
  -device vfio-pci,host=<device-BDF>,id=vfio_dcu6,bus=rp6 \
  -device vfio-pci,host=<device-BDF>,id=vfio_dcu7,bus=rp7 \
  -device vfio-pci,host=<device-BDF>,id=vfio_dcu8,bus=rp8 \
  -device vfio-pci,host=<device-BDF>,id=dev_10_00 -device vfio-pci,host=<device-BDF>,id=dev_13_00 \
  -device vfio-pci,host=<device-BDF>,id=dev_16_00 -device vfio-pci,host=<device-BDF>,id=dev_19_00 \
  -device vfio-pci,host=<device-BDF>,id=dev_1c_00 -device vfio-pci,host=<device-BDF>,id=dev_1f_00 \
  -device vfio-pci,host=<device-BDF>,id=dev_20_00 \
  -fw_cfg name=opt/ovmf/X-PciMmio64Mb,string=1048576
```
> 需设 CSV3 预留内存：`grubby --args="csv3_guest_rsv_mb=16G"`，`cat /sys/kernel/debug/gpu_extra/shmem_info` 查预留情况。

### 6.4 虚拟机内装驱动 + 推理
```bash
wget https://download.sourcefind.cn:65024/file/6/csv-tee驱动/dtk-26.04-tee驱动/rock-6.3.32-V1.6.0.run
chmod +x rock-6.3.32-V1.6.0.run && ./rock-6.3.32-V1.6.0.run
/opt/hyhal/bin/hy-smi   # 查卡
# docker（用 podman 例）：
podman run -it --name lzh --network=host --privileged -v /models:/models -v /opt/hyhal:/opt/hyhal \
  --device=/dev/kfd --device=/dev/dri --device=/dev/mkfd --ipc=host --group-add video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  <internal_registry>/dcu/admin/base/custom:deepseek-v4-20260615 bash
```
> DS-V4-Flash SGLang 启动脚本（含 DCU 优化 env：`ROCBLAS_TENSILE_LIBPATH`/`HIP_KERNEL_BATCH_CEILING`/`GPU_MAX_HW_QUEUES`/`USE_DCU_CUSTOM_ALLREDUCE`/`SGLANG_ROCM_USE_AITER_MOE`/`SGLANG_USE_FP8_W8A8_MOE`/`SGLANG_DSV4_MODE=2604` 等）—— 详见来源文档 §六 启动脚本，按实际模型路径调整。

### 6.5 驱动更新 + 工单
- 社区驱动：[dtk-26.04-tee 驱动](https://download.sourcefind.cn:65024/6/main/csv-tee%E9%A9%B1%E5%8A%A8/dtk-26.04-tee%E9%A9%B1%E5%8A%A8)
- 参考 ontrack 工单（需权限）：<internal-ticket>（BW1000 CSV3 部署 deepseek_v4 报错）、<internal-ticket>（CSV3 加密虚拟机创建失败）、<internal-ticket>（CSV2 直通失败）

## 7. 性能影响结论（吸收 ckvg5AWIMkdk，数据表完整收录）

> 四场景：①裸机关 ACS/IOMMU ②裸机开 ACS/IOMMU ③虚拟机直通 ④加密虚拟机（均开 ACS/IOMMU）。测试：BW1000，8 卡，DeepSeek-V4-Flash-INT8-Channel，sglang，bs=1/4/8/16/32。

**核心结论**：
- **BW1000 HYswitch 架构优势**：卡间互联（HSL）带宽 **310GB/s 不受 ACS/IOMMU 和虚拟化影响**，从根本上解决 K100_AI 的 NBIO 瓶颈（K100_AI 开 ACS/IOMMU 时 D2D 被限 38GB/s）
- **虚拟机直通对 PCIe 带宽降约 17%**（最低 25GB/s）；对推理性能与并发强相关：低并发（bs=1）降 **14%-18.6%**，高并发（bs≥16）降 **1.2%-7.1%**
- **裸机开 ACS/IOMMU 对 BW1000 几乎无影响**（降幅 <2.5%），而 K100_AI 同等配置降达 25.4%
- **加密虚拟机性能代价可控**：高并发（bs≥16）损失 **<2%**，适合高安全+高并发生产环境；低并发（bs=1）最大 22.4%
- **部署建议**：裸机可按安全需求灵活开/关 ACS/IOMMU；虚拟机直通建议高并发（bs≥8）；虚拟化 AI 推理优先 BW1000

> **明细数据表（表4-8）已按 skill 内容纪律剥离**：D2D 带宽、ACS/IOMMU 推理降幅、四场景 GenerateThroughput/TotalThroughput 逐行 tokens/s 等具体数字属性能基线，不入本 skill。如需引用，溯源：内部文档（已脱敏）。
> - 测试数据 xlsx：内部文件（已脱敏）
> - 测试结果 csv：内部文件（已脱敏）
> - 性能分析文档 file_id：<file_id>
>
> 定性要点（供部署决策）：K100_AI 仅当 ACS 与 IOMMU 同时关闭时 D2D 才达峰值（否则限 38GB/s，NBIO 瓶颈）；BW1000 卡间互联 HSL 310GB/s 不受 ACS/IOMMU 与虚拟化影响；虚拟机直通对 PCIe 带宽降约 17%、对推理低并发降 14-18.6%/高并发降 1.2-7.1%；加密虚拟机高并发损失 <2%。

## 8. 子文档溯源（重要）

| 子文档 | file_id | 状态 | 处理 |
|---|---|---|---|
| 主手册《DCU安全：加密虚拟机直通速成手册》 | `<file_id>` | 可达 | 全文吸收 |
| 《探究BW1000下虚拟机直通对DCU性能带来的影响》 | `<file_id>` | 可达 | 全文吸收 + 数据表4-8收录 |
| 《虚拟化直通速成手册》（原作者，明文直通速成） | `<file_id>` | 可达 | 吸收：明文直通最简命令(§3.3.1)、fixup header 详细流程+开机自启(§2.6.1)、明文平台 bridge 限定(§2.7)、VM 内装原厂驱动(§4.1.1)、VM 网络/virt-copy-in 传文件踩坑(§4.3) |
| 测试数据 xlsx | `<file_id>` | CLI read-file 失败 | 仅留 URL 溯源 |
| 测试结果 csv | `<file_id>` | CLI read-file 失败 | 仅留 URL 溯源 |

## 9. 与兄弟 skill 关系
- `dcu-docker`：虚拟机内 docker 容器部署（本文用 TEE 镜像，与其互补）
- `dcu-env` / `dcu-topology`：BIOS/IOMMU/PCIe group 检测（本文 §2.5/§2.6 依赖）
- `dcu-rccl-test`：多卡 P2P 坑（`NCCL_P2P_DISABLE=1`、ACS 致 P2P 受阻，本文 §5 踩坑与之呼应）
- `dcu-kernel-dev`：DCU 算子（虚拟机内运行的算子底层）

## 使用须知
- 主文档 39 个 + 分析文档 15 个 = 54 个文档截图占位（BIOS 设置、启动截图、仪表盘等）已全部剔除，不搬运二进制
- 性能数据表为来源文档实测（BW1000/X7950 + K100_AI），作为架构选型参考；具体数字随硬件/驱动/模型版本变化，**部署前以实测为准**，不视为固定基线
- 高危操作（BIOS/kvm-amd.sev/mem_encrypt/VFIO 解绑 hycu/加密虚拟机）谨慎，优先测试集群验证
- ontrack 工单链接需内网权限，仅作溯源
