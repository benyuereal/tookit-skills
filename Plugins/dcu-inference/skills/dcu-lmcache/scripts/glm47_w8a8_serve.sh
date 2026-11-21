#!/usr/bin/env bash
# GLM4.7-W8A8 单机 8 卡推理启动（LMCache NVMe offload，长/超长上下文场景）
# 来源：海光 DCU LMCache 最佳实践指南主文档实测脚本（已归一化全角空白）
# 关键纪律：
#  - 量化方法 -q slimquant_marlin（W8A8）
#  - VLLM_NUMA_BIND + 各 RANK NUMA 绑核（数值按实际拓扑）
#  - PYTHONHASHSEED=0 保证 KV 缓存键一致，复用缓存
#  - LMCACHE_CONFIG_FILE 指向 NVMe offload 配置
set -a
export VLLM_TORCH_PROFILER_DIR=/prof
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export NCCL_P2P_NVL_CHUNKSIZE=131072
export VLLM_RPC_TIMEOUT=1800000
export VLLM_NUMA_BIND=1
export VLLM_RANK0_NUMA=3
export VLLM_RANK1_NUMA=1
export VLLM_RANK2_NUMA=1
export VLLM_RANK3_NUMA=0
export VLLM_RANK4_NUMA=7
export VLLM_RANK5_NUMA=5
export VLLM_RANK6_NUMA=5
export VLLM_RANK7_NUMA=4
export VLLM_USE_GLOBAL_CACHE13=1
export VLLM_FUSED_MOE_CHUNK_SIZE=8192
export VLLM_ZERO_OVERHEAD=1
export W8A8_SUPPORT_METHODS=3
export VLLM_USE_PD_SPLIT=1
export LMCACHE_CONFIG_FILE=./lmcache_nvmeoffload.yaml
export LMCACHE_USE_EXPERIMENTAL=True
export PYTHONHASHSEED=0
set +a

vllm serve "/public/models/GLM-4.7-W8A8" \
  -q slimquant_marlin \
  --trust-remote-code --dtype bfloat16 \
  --max-model-len 202752 \
  -tp 8 \
  --gpu-memory-utilization 0.95 \
  --disable-log-requests \
  --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
