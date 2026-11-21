#!/bin/bash
# 曙光 SHCA IB BW1000 双机 TP16 vLLM 启动参考（GLM-5.2-int4 + MTP）
# 来源：内部文档 <internal-doc-link>《曙光IB BW1000双机RMDA调优记录》（ontrack <internal-ticket>）
# 用法：每节点容器内执行；NCCL_TOPO_FILE 必须指向本节点自己的 topo 文件（GUID 各节点不同，
#       在 Ray 启动脚本中按 ib0 IP 区分节点动态设置，勿用头节点广播值）。
source /opt/dtk/env.sh
# --- SHCA 官方 RCCL 网络插件（关键：不要用通用 verbs 桥接库）---
export LD_LIBRARY_PATH=/opt/hpc/software/app/rccl/shca_rdma_plugins/v8/lib:$LD_LIBRARY_PATH
export NCCL_NET_PLUGIN=/opt/hpc/software/app/rccl/shca_rdma_plugins/v8/lib/librccl-net-shca.so
# --- Ray 环境传播（确保 NCCL_/GLOO_ 等 env 传到 Ray worker）---
export VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY=GLOO_,HSA_,LD_,RAY_,NCCL_
export HSA_FORCE_FINE_GRAIN_PCIE=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1
# --- 节点绑定：VLLM_HOST_IP 取本节点 ib0 IP（确保 Ray 内部通信绑定 IB 接口）---
export VLLM_HOST_IP=$(ip -4 addr show ib0 | awk '/inet/ {print $2}' | cut -d/ -f1)
# --- Topo：本节点自己的 topo 文件（hy-smi --showtopo 取实际 GUID 生成，修正 vendor 字段）---
export NCCL_TOPO_FILE=/data/bash/rdma_setup/topo_m06r4n14.xml
# --- OOB 与 Socket ---
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
# --- HCA 枚举（注意：不加 :端口号 后缀，该版本 RCCL 不接受）---
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=shca_0,shca_1,shca_2,shca_3
# --- 兼容性（SHCA 不支持 PCIe Relaxed Ordering / ECE）---
export NCCL_IB_PCI_RELAXED_ORDERING=0
export NCCL_IB_ECE_ENABLE=0
export NCCL_IB_QPS_PER_CONNECTION=2
# --- 通道数两端显式锁死（避免自动协商不一致导致消息截断）---
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export NCCL_NCHANNELS_PER_NET_PEER=4
# --- GDR（DMABUF 是前提；NET_GDR_READ 配合该插件实测需置 0）---
export NCCL_DMABUF_ENABLE=1
export NCCL_NET_GDR_READ=0
unset NCCL_NET_GDR_LEVEL
# --- 调试（验证终态后建议降级）---
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
# --- 设备与 vLLM ---
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_SPEC_DECODE_EAGER=1
export VLLM_MLA_DISABLE=0
export VLLM_USE_FLASH_MLA=1
export VLLM_RPC_TIMEOUT=1800000
export VLLM_LOGGING_LEVEL=INFO
export VLLM_NUMA_BIND=1
export VLLM_RANK0_NUMA=0
export VLLM_RANK1_NUMA=1
export VLLM_RANK2_NUMA=2
export VLLM_RANK3_NUMA=3
export VLLM_RANK4_NUMA=4
export VLLM_RANK5_NUMA=5
export VLLM_RANK6_NUMA=6
export VLLM_RANK7_NUMA=7
export LMSLIM_USE_GLOBAL_MOE_CACHE=1
export VLLM_USE_RAY_COMPILED_DAG=0

vllm serve \
  --model /data/model/GLM-5.2-INT4-w4a8/ \
  -q slimquant_w4a8 \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-model-len 1024 \
  --port 8001 \
  -tp 16 \
  --distributed-executor-backend ray \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 16 \
  --block-size 256 \
  --kv-cache-dtype fp8_ds_mla \
  --served-model-name glm-4.2 \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --max-num-batched-tokens 32768 \
  --enable-prefix-caching \
  --speculative-config '{"num_speculative_tokens": 1, "method": "mtp"}'

# 验证终态（NCCL_DEBUG=INFO 下）：
#   NET/IBext_v8 : Using [0]shca_0:<ib0_ip>
#   Channel 00/0 : 8[9000] -> 0[9000] [send] via NET/IBext_v8/0/GDRDMA
#   p2pnChannels=16, perPeer=4；topo 无 "cannot be found" 警告
#   /v1/models 返回模型名，Application startup complete
