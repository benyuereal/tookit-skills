#!/usr/bin/env bash
# Qwen3-32B 单机 4 卡推理启动（LMCache CPU/NVMe offload，长上下文 16k-1k 场景）
# 来源：海光 DCU LMCache 最佳实践指南主文档实测脚本（已归一化全角空白）
# 关键纪律：
#  - HIP_VISIBLE_DEVICES 指定卡（从 0 起，不靠 -g N 挑卡）
#  - VLLM_NUMA_BIND + 各 RANK NUMA 绑核（hycpu 环境必设，数值按实际拓扑）
#  - PYTHONHASHSEED=0 保证相同文本生成的 KV 缓存键一致，实现缓存复用
#  - LMCACHE_CONFIG_FILE 指向 offload 配置（见 scripts/lmcache_config_*.yaml）
set -a
export ALLREDUCE_STREAM_WITH_COMPUTE=1
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export VLLM_PCIE_USE_CUSTOM_ALLREDUCE=1   # K100-AI 需要
# hycpu 环境建议绑核（数值按实际 NUMA 拓扑调整）
export VLLM_NUMA_BIND=1
export VLLM_RANK0_NUMA=3
export VLLM_RANK1_NUMA=1
export VLLM_RANK2_NUMA=1
export VLLM_RANK3_NUMA=0
export VLLM_RANK4_NUMA=7
export VLLM_RANK5_NUMA=5
export VLLM_RANK6_NUMA=5
export VLLM_RANK7_NUMA=4
export HIP_VISIBLE_DEVICES=0,1,2,3
export PYTHONHASHSEED=0   # 确保 KV 缓存键一致，复用缓存
# LMCache 环境变量
export LMCACHE_CONFIG_FILE=./lmcache_nvmeoffload.yaml
export LMCACHE_USE_EXPERIMENTAL=True
set +a

vllm serve /data/models/Qwen3-32B \
  --trust-remote-code -tp 4 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.95 \
  --enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'
