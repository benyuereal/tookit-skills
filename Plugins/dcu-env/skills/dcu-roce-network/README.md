# dcu-roce-network — 海光 DCU 集群 RoCEv2 网络配置

多机（4 机及以上）DCU 集群的 RoCEv2 物理网络搭建 skill：IP/MTU 规划、网卡 MAC 收集、
**静态 ARP 全量注入（关键）**、QoS/PFC/ECN 配置、RDMA 连通性验证。

## 核心内容

- RoCEv2 网络搭建完整流程（5 步：IP → MAC → 静态 ARP → QoS → 连通性验证）
- 静态 ARP 全量注入的原理与价值（防 ARP 欺骗 + 多网卡容错）
- QoS / PFC / ECN / TOS 无损以太网配置
- 与 dcu-rccl-test 的分层：本 skill 是「网络前置层」，rccl-test 是「上层通信测试」

## 状态

✅ 已完成整理并验证（脚本 bash -n 全部通过）。来源为《四机 BW1100 RoCE v2 网络配置方案》
（工行项目）。具体 IP/接口名为案例值，复用前需按实际环境改写。

详见 `SKILL.md`。

## 衔接

- 上层集合通信测试：dcu-rccl-test（网络建好后跑 RCCL allreduce/alltoall）
- 拓扑/绑核：dcu-topology
- 环境探测：dcu-env
