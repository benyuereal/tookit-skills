---
name: dcu-roce-network
description: 海光 DCU 集群 RoCEv2 网络环境搭建：IP/MTU 规划、静态 ARP 全量注入（关键）、QoS/PFC/ECN 配置、RDMA 连通性验证。RCCL 多机测试/训练的前置物理网络层。
version: 1.0.0
author: DCU Engineer
license: MIT
---

# DCU 集群 RoCEv2 网络配置

## Overview

本 skill 覆盖**多机 DCU 集群的 RoCEv2 物理网络如何搭建**——从 IP/MTU 规划、网卡 MAC 收集、**静态 ARP 全量注入（关键）**、到 QoS/PFC/ECN 配置与 RDMA 连通性验证。这是上层集合通信（RCCL allreduce/alltoall）测试与分布式训练能跑通、能拿到高带宽的**物理网络前置层**。

> 📌 **与 dcu-rccl-test 的分层**：
> - **本 skill（dcu-roce-network）**：把 RoCE 网络"搭起来"——IP/MAC/ARP/QoS 这些物理链路层配置，让多机之间真正能 RDMA 互通。
> - **dcu-rccl-test**：在已搭好的网络上跑集合通信性能测试与调优（NCCL_IB_HCA、topo mapping、xdp_size、NCCL_IB_TC 等软件层）。
> - 两者是「前置网络 vs 上层通信测试」的上下游关系，多机测试前若网络没搭对，RCCL 跑不出高性能。

## When to Use

- 多机 DCU 集群（4 机及以上）要跑 RCCL 集合通信 / 分布式训练，需要先把 RoCEv2 网络配好
- 多机 RCCL 测试带宽不达标，需排查是否是**物理网络层**（ARP/QoS/PFC/MTU）问题，而非软件层 topo 问题
- 新集群（如 BW1100、BW1000 等 OAM 机型 + 200G NDR 网卡）首次组网

Don't use for：单机多卡（无跨机网络）；已有网络仅需上层 RCCL 调优（见 dcu-rccl-test）。

## 适用硬件（本文案例）

| 项 | 配置 | 说明 |
|---|---|---|
| 服务器 | Hygon 7490 ×2 / DDR5 5600 32G×24 / BW1100 OAM×1 | 共 4 台（node1~4） |
| 高速网卡 | 双端口 200G NDR HCA 卡 ×4（Mellanox MCX755106AS-HEAT 2-port 200G QSFP112） | 每节点 8 个 RDMA 网口 |
| 业务网卡 | 双口 I350 1G + 双口 25G（含光模块） | 管理/业务网（非 RDMA） |
| 交换机 | 华三 S12504CR-1（框式，支持 RoCE） | 1 台 |

> 核心点是**高速 NDR 网卡的多口（8 口）组网**——这是 RoCEv2 高带宽多机通信的基础。不同机型/网卡同理，只是接口名与设备数不同。

## 核心经验流程（按序执行）

```
1. IP 规划与配置   → setup-network.sh        （静态 IP + MTU 9000 + 自启）
2. MAC 收集       → collect_cluster.sh       （生成 cluster_topology.conf）
3. 静态 ARP 注入  → apply_arp.sh             （关键！！防 ARP 欺骗/多网卡容错）
4. QoS 配置       → set_roce_qos.sh          （Trust DSCP + PFC + ECN + TOS）
5. 连通性验证     → rdma_connectivity_test.sh （同/跨子网 ping + ib_write_bw）
```

每步的脚本在 `scripts/` 下，均需按本节点**实际网口名**改写（勿照抄 p66p1 等，详见踩坑）。

## IP 地址规划

- **网段**：`124.254.0.0/24`，交换机侧网关 `124.254.0.254/24`（本文案例的具体网段；新集群按实际规划替换，**方法通用**）
- **划分**：8 个 `/27` 子网，每个对应一个物理网口。`net.ipv4` 子网掩码为 `/27`
- **8 个网口各自独立子网**：`p66p1`/`p66p2`/`p61p1`/`p61p2`/`p82p1`/`p82p2`/`p72p1`/`p72p2`
- 4 机在同一网口上处于同一 `/27` 子网（如 p66p1 子网 = node1~4 的 `.1~.4`）

> ⚠️ **IP 与 MAC 必须一一对应、核对清楚**（文档标注的"非常重要"踩坑之一）。子网内 node 偏移地址要精确，否则静态 ARP 注入时会映射错目标。

## 关键点一：静态 ARP 全量注入（apply_arp.sh）

这是本文标注「**关键！！非常重要！！**」的部分，务必理解其价值：

**做了什么**：在每台节点上，读取 `cluster_topology.conf`，比对「本机 MAC」自动过滤本机 IP，然后把**远端所有节点的 IP/MAC 全量注入到本机的每一个网口**（写入永久 permanent 邻居条目）。

**为什么必须静态注入**：
1. **避免 ARP 广播泛滥与欺骗**：内核设了 `arp_ignore=2`、`rp_filter=2` 限制 ARP 响应逻辑；静态 `nud permanent` 彻底关闭链路上的 ARP 广播请求，数据包拿到 IP 即可直接装配 MAC 发送，无需动态 ARP。
2. **多路径/网卡容错与多网卡协同**：节点间存在多条链路（如 p66p1→p72p2 共 8 网口）。由于路由表或应用可能从本机**任意网口**发包到远端**任意 IP**，必须让本机每个网口都持有远端所有 IP 的 ARP 映射，否则任何接口缺邻居条目都会导致发包丢弃或延迟。

**验证**：
```bash
ip neigh show
ip neigh show dev <任一同子网网口> | grep -i permanent | wc -l
```

⚠️ **重启后 ARP 配置失效**，需重新执行 `apply_arp.sh`（建议做成 systemd 自启，同 setup-network 做法）。

## 关键点二：QoS / PFC / ECN 配置（set_roce_qos.sh）

RoCEv2 必须在交换机与网卡双侧配合启用无损以太网能力，否则拥塞丢包会严重拉低 RDMA 带宽。本脚本策略：

| 项 | 配置 | 作用 |
|---|---|---|
| Trust 模式 | `mlnx_qos --trust dscp` | 信任 DSCP 标记 |
| PFC | 队列 3、6 = 1（`--pfc 0,0,0,1,0,0,1,0`） | 无损队列（优先级流控） |
| 链路流控 | `ethtool -A tx off rx off` | 关闭网卡全局链路层流控 |
| TOS | `cma_roce_tos -t 96` | 设置 RoCE 流 TOS |
| ECN | 优先级 3、6 使能 `roce_np`/`roce_rp` | 显式拥塞通知 |

> ⚠️ **交换机侧也必须同步配置**：MTU（9000）、PFC（对应队列）、QoS/DSCP 策略均需与网卡侧一致。
> **TOS 与 NCCL_IB_TC 是两层参数**：本脚本 `cma_roce_tos -t 96` 设的是 RDMA 流底层 TOS（本文案例值 96）；
> 上层应用层 `NCCL_IB_TC`（华三交换机一般 162 / cumulus 默认 96，见 dcu-rccl-test）需与该 QoS 队列映射对应，**二者要能对上**，否则流量走错优先级队列、PFC 不生效，拥塞时丢包拉低带宽。

## 关键点三：连通性验证（rdma_connectivity_test.sh）

网络配好后，逐档验证确认真正互通再进上层 RCCL：

1. **同子网点对点 ping**：`bash rdma_connectivity_test.sh ping-same`（8 子网两两互 ping，自动统计失败链路）
2. **跨子网 ping**：`ping <目标IP> -I <源IP>`（指定源接口）
3. **ib_write_bw 带宽**：对端先起服务端，客户端执行（`-x 3` 为 RoCE 传输类型，`-s 65536` 测大包带宽）

示例：
```bash
# 服务端
ib_write_bw -d mlx5_2 -a -F -x 3 -q 1 -s 65536 --report_gbits
# 客户端
ib_write_bw -d mlx5_3 -a -F -x 3 -q 1 -s 65536 --report_gbits 124.254.0.66
```

## systemd 自启（IP 与 QoS 均建议）

```bash
# 1) IP 配置自启
cat <<'EOF' > /etc/systemd/system/node-network.service
[Unit]
Description=Set Static IP and MTU 9000 for Network Interfaces
After=network.target
[Service]
Type=oneshot
ExecStart=/usr/local/bin/setup-network.sh
RemainAfterExit=yes
[Install]
WantedBy=multi-user.target
EOF

# 2) QoS 配置自启
cat <<'EOF' > /etc/systemd/system/roce-qos.service
[Unit]
Description=Configure RoCE QoS (PFC, ECN, TOS)
After=network-online.target
Wants=network-online.target
[Service]
Type=oneshot
ExecStart=/usr/local/bin/set_roce_qos.sh
RemainAfterExit=yes
[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now node-network.service roce-qos.service
```

> ARP 配置也应纳入自启（重启后会失效）。脚本需先 `chmod +x` 置 `/usr/local/bin/`。

## 踩坑（重要）

1. **IP 与 MAC 一一对应、核对清楚** —— 运行 `collect_cluster.sh` 前，先在所有节点执行 `ip -br link` 确认 NDR 网卡物理接口名。务必用 `ibstat`/`ibv_devinfo` 或 `ls -l /sys/class/net/<iface>/device/infiniband` 核对 `mlx5_x` 与系统网卡名（如 p66p1）的对应关系。
2. **远端节点 IP/MAC 必须全量注入到本机每个网口** —— 缺任一接口的邻居条目会导致发包丢弃或延迟（见「关键点一」）。
3. **MTU 必须与交换机一致且设为 9000** —— IP 脚本中 `mtu 9000`，交换机侧同步。
4. **重启后 ARP/QoS 配置失效** —— 需重跑脚本或做成 systemd 自启。
5. **mlnx_qos / cma_roce_tos 需 Mellanox OFED 工具** —— 若报命令不存在，安装 OFED 或在对应网卡工具环境执行。
6. **接口名不可照抄** —— 本文示例 p66p1/p66p2/... 是具体机型（BW1100 + 200G NDR）的实际名，不同机型/网卡接口名不同，务必按实际改写脚本中的 `IFACES` 与 IP_MAP。

## 与 RCCL 测试的衔接

网络建好后，进上层 RCCL 测试（见 `dcu-rccl-test`）时需注意：
- **xdp_size 开启**（驱动 ≥ 6.3.30）：`hy-smi --setdriverparams xdp_size=15` + `hy-smi --loaddriver`
- **topo mapping 文件生成**：用内部 topo_tools 生成 `topo_mapping_custom.xml`，`export NCCL_TOPO_MAPPING_FILE=...`
- **NCCL_IB_HCA** 按实际 `mlx5_x` 填、**NCCL_IB_TC**（华三 162 / cumulus 96）与 QoS 队列对应
- 若 RCCL 带宽不达标，先回到本 skill 排查网络层（ARP 是否全量注入、PFC/ECN 是否生效、MTU 是否一致），再查 topo/驱动层。

## Scripts

| 脚本 | 用途 |
|---|---|
| `scripts/setup-network.sh` | 节点静态 IP + MTU 9000（node1~4 参数化） |
| `scripts/collect_cluster.sh` | 收集集群各节点网口 IP/MAC → cluster_topology.conf |
| `scripts/apply_arp.sh` | 内核 ARP 参数 + 远端 IP/MAC 静态全量注入（关键） |
| `scripts/set_roce_qos.sh` | RoCE QoS/PFC/ECN/TOS 配置 |
| `scripts/rdma_connectivity_test.sh` | 同/跨子网 ping + ib_write_bw 连通性验证 |

> 所有脚本均按本文案例 IP/接口名固化，**使用前必须按实际环境改写**再执行。

## References

- 来源：四机 BW1100 RoCE v2 网络配置方案（工行项目，2026-08-18 初版配置副本）
- 关联 skill：
  - `dcu-rccl-test`（上层 RCCL 测试 + NCCL 调优 env）
  - `dcu-topology`（卡↔NUMA/拓扑采集）
  - `dcu-env`（环境探测）、`dcu-ssh`（远程连接）
- RCCL 通信测试补充文档：内部文档《rccl-test 通信测试-260610》
