# 测试环境与 NCCL·RCCL 调优 env 速查

> 通用测试指令、单机调优 env、alltoall 必做 unset、集群 mpirun 模板。
> 2026-10-16 自 `env/dcu-rccl-test/SKILL.md` 下沉（SKILL.md 瘦身到 500 行内），内容原文照录。

## 测试环境 / NCCL·RCCL 调优 env 速查

> 内部手册给出的各机型"最佳性能"对应的驱动 / 镜像 / 指令，提炼如下（实测请以"输出规范"原样汇报）。

### 通用测试指令（单机）
```bash
# 通用（清 NCCL_NCHANNELS_PER_PEER 后）
./build/all_reduce_perf -b 2 -e 2G -f 2 -g <N>     # N=2/4/8
./build/alltoall_perf  -b 2 -e 2G -f 2 -g <N>
```
- 驱动镜像（多数机型基线）：`<internal_registry>/dcu/admin/base/vllm:<tag>`
- BW1100 基线镜像：`...vllm:0.11.0-ubuntu22.04-dtk26.04-0130-py3.10-20260204`

### 单机调优 env（BW150 / BW1000 / BW1100 提速关键）
```bash
# BW150（4卡互联拓扑，驱动 6.3.22-V1.2.0）2卡/4卡/8卡最佳 env：
NCCL_SIMPLE_CHANNELS=20 RCCL_P2P_XHCL_CHANNEL_NUM=16 RCCL_COLL_XHCL_CHANNEL_NUM=16 ./build/all_reduce_perf -b 2 -e 2G -f 2 -g <N>
# BW150 4卡额外（驱动同）指定 ring 算法：
NCCL_RINGS="0 1 2 3|0 3 2 1|0 1 3 2|0 2 3 1|0 2 1 3|0 3 1 2" NCCL_ALGO=Ring NCCL_SIMPLE_CHANNELS=20 RCCL_P2P_XHCL_CHANNEL_NUM=16 RCCL_COLL_XHCL_CHANNEL_NUM=16 ./build/all_reduce_perf -b 2 -e 2G -f 2 -g 4

# BW1000（X7950H0 OAM，驱动 6.3.16-V1.1.0a）8卡最佳 env：
NCCL_SIMPLE_CHANNELS=32 RCCL_P2P_XHCL_CHANNEL_NUM=31 RCCL_COLL_XHCL_CHANNEL_NUM=28 ./build/all_reduce_perf -b 2 -e 2G -f 2 -g 8

# BW1100（驱动 6.3.31-V1.5.0c）8卡最佳（无额外 channel env）：
./build/all_reduce_perf -b 2 -e 2G -f 2 -g 8
```

### ⚠️ alltoall 单机双卡必做：unset NCCL_NCHANNELS_PER_PEER
```bash
# BW150 / BW1000 2卡 alltoall 最佳前必须先清掉镜像可能预置的 NCCL_NCHANNELS_PER_PEER
unset NCCL_NCHANNELS_PER_PEER
NCCL_SIMPLE_CHANNELS=32 RCCL_P2P_XHCL_CHANNEL_NUM=31 RCCL_COLL_XHCL_CHANNEL_NUM=28 ./build/alltoall_perf -b 2 -e 2G -f 2 -g 2
```

### 集群（BW1000 X7950H0 OAM + 200G IB）mpirun 模板
```bash
# 驱动 6.3.16-V1.1.0a；镜像 <internal_registry>/dcu/admin/base/vllm:<tag>
# 16卡(-H master:8,node1:8) / 32卡(-H master:8,node1:8,node2:8,node3:8)
mpirun --prefix /opt/mpi -np <N> -H <hostlist> \
  --allow-run-as-root --mca btl_tcp_if_include p14p2 --mca plm_rsh_args "-p 1777" \
  -x NCCL_SOCKET_IFNAME=p14p2 -x HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 -x ROCM_PATH -x LD_LIBRARY_PATH \
  -x NCCL_IB_HCA=mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_7,mlx5_8,mlx5_9,mlx5_10 \
  -x NCCL_NET_GDR_LEVEL=SYS -x NCCL_NET_GDR_READ=1 \
  ./build/all_reduce_perf -b 2 -e 2G -f 2 -g 1
# alltoall 仅末行换 alltoall_perf
```
> 注：上述 `p14p2` 为管理/IB 网口名、`mlx5_*` 为 IB HCA——这些须按实际环境替换。
> **`-p 1777` 是容器 sshd 端口**，取决于测试容器启动时所指定的 SSH 端口（见 dcu-docker 容器免密：起容器时 sshd 监听端口 + mpirun `--mca plm_rsh_args "-p <端口>"` 必须一致）。手册示例用 1777，你的容器若用 2345 或其他端口，这里同步改。物理机测试（非容器）则无需 `--mca plm_rsh_args`，且 `NCCL_IB_HCA` 按物理机实际网卡填。
