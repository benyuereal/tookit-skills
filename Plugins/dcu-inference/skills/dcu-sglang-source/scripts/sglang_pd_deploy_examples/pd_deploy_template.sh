#!/usr/bin/env bash
# ============================================================================
# SGLang PD 分离（disaggregation）部署模板 —— DCU 实战（脱敏版）
#
# 来源：某 BW1100 项目 PD 分离部署方案（SGLang 0.5.12 海光 fork + DTK 26.04 + DeepEP）。
# 本文件为**脱敏参考模板**：IP / 模型路径 / 镜像地址 / wheel 名 / 客户标识均已去除。
# 使用前把下方 `${VAR}` 全部替换/export 为实际值，再按需成为可执行的启动脚本。
# ⚠ 版本锚点：SGLang 0.5.12 海光 fork + DTK 26.04；其他版本参数名/默认值可能不同。
#
# 说明：环境变量是否生效以目标镜像实际为准（对照 dcu-sglang-source §二/§三·补 源码落点）。
# ============================================================================

# ↓↓↓ 使用前必须替换的占位变量 ↓↓↓
export P_IP=            # prefill 节点 IP
export D_IP=            # decode 节点 IP
export P_PORT=          # prefill 服务端口
export D_PORT=          # decode 服务端口
export P_INIT_PORT=     # prefill dist-init 端口
export D_INIT_PORT=     # decode dist-init 端口
export D_MASTER_IP=     # decode 多节点主节点 IP
export ROUTER_PORT=     # router 端口
export MODEL_PATH=      # 模型路径（本地目录）
export MODEL_NAME=      # served-model-name
export IB_DEVICES=      # 如 mlx5_2:1,mlx5_3:1,...（:1=端口1）
export MGMT_IF=         # 管理网口，如 enp113s0f0np0 / p14p1
export TOPO_FILE=       # topo.config 路径（卡网映射）
export EP_CONFIG_JSON=  # DeepEP ep_config.json 路径
export KV_DTYPE=        # 如 bfloat16 / fp8_e4m3
export QUANT=           # 如 slimquant_marlin
export CTX_LEN=         # 如 131072

# ============================================================================
# 一、关键环境变量（PD 分离 + DeepEP + MoE 融合 + 多模态通用）
# ============================================================================
# 把这些放入启动 shell（P/D/单机各节点按需取用）。

# --- 网络 / HCA ---
export NCCL_SOCKET_IFNAME="${MGMT_IF}"
export GLOO_SOCKET_IFNAME="${MGMT_IF}"
export UCX_NET_DEVICES="${IB_DEVICES}"
export NCCL_IB_HCA="${IB_DEVICES}"
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=16
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   # 按需指定，编号从 0 起

# --- ROCShmem（DeepEP / RDMA 关键）---
export ROCSHMEM_IB_GID_INDEX=0
export ROCSHMEM_TOPO_FILE_FORCE="${TOPO_FILE}"
export ROCSHMEM_HEAP_SIZE=3173741824
export ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=288
export ROCSHMEM_ALLOWED_IBV_DEVICES="${IB_DEVICES}"
export ROCSHMEM_MAX_NUM_CONTEXTS=60          # 按需

# --- HIP / HSA / GPU 队列（decode/大并发调优）---
export GPU_MAX_HW_QUEUES=2                    # decode 常用 3
export HIP_GRAPH_ACCUMULATE_DISPATCH=1
export HIP_H2D_DIRECT_COPY_THRESHOLD=32768
export HIP_H2D_DISABLE_COPY_BUFFER=0
export HIP_D2H_DIRECT_COPY_THRESHOLD=512
export HIP_D2H_HSAAPI_COPY_THRESHOLD=512
export HIP_KERNEL_BATCH_CEILING=100
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export HSA_KERNARG_POOL_SIZE=8388608
export ROC_AQL_QUEUE_SIZE=131072
export GLIBC_TUNABLES=glibc.rtld.optional_static_tls=0x40000

# --- MC / 自定义 allreduce（PD/通信）---
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export MC_TE_FILTERS="${IB_DEVICES}"
export USE_DCU_CUSTOM_ALLREDUCE=1
export ALLREDUCE_STREAM_WITH_COMPUTE=1

# --- SGLang 融合算子 / 量化开关（DCU 适配核心）---
export SGLANG_ENABLE_SPEC_V2=1                # MTP/投机 v2
export SGLANG_ROCM_USE_AITER_MOE=1
export SGLANG_ROCM_USE_AITER_TILELANG_MHC=1
export SGLANG_USE_DEEPGEMM_MOE=1
export SGLANG_USE_FP8_W8A8_MOE=1
export SGLANG_USE_LIGHTOP=1
export SGLANG_USE_LIGHTOP_GROUP_FP8_QUANT=1
export SGLANG_USE_FUSED_MLA_CAT=1
export SGLANG_USE_FUSED_DPSKV4_SILU_MUL_FP8_QUANT=1
export SGLANG_USE_FUSED_DPSKV4_QNORM_ROPE_KV_ROPE_QUANT=1
export SGLANG_USE_LINEAR_BF16_FP32_USE_BLASLT=1
export SGLANG_OPT_USE_FUSED_HASH_TOPK=true
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_APPLY_CONFIG_BACKUP=none
export SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=0
export SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA=0
export SGLANG_OPT_FLASHMLA_SPARSE_PREFILL=1    # decode 常用 0
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=256   # decode 常用 128
export SGLANG_KVALLOC_KERNEL=1
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
export SGLANG_USE_MODELSCOPE=1
export SGLANG_SET_CPU_AFFINITY=1

# --- PD 分离超时（disaggregation）---
export SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=1200
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=1200

# ============================================================================
# 二、Prefill 节点启动（--disaggregation-mode prefill）
# ============================================================================
# 适合长输入、输出相对短、不要求低 TTFT；P 副本内部常用 CP8+EP8。
# 此段为参考命令，复制到目标机前先替换 ${...}/布尔开关按需调整。
#   sglang serve "${EXTRA_OPT}" \
#     --model-path "${MODEL_PATH}" --trust-remote-code \
#     --served-model-name "${MODEL_NAME}" \
#     --host "${P_IP}" --port "${P_PORT}" \
#     --dist-init-addr "${P_IP}:${P_INIT_PORT}" --nnodes 1 --node-rank 0 \
#     --tp-size 8 --dp-size 1 --pp-size 1 \
#     --attn-cp-size 8 --ep-size 8 \
#     --moe-dense-tp-size 1 \
#     --moe-a2a-backend deepep --deepep-mode normal \
#     --enable-nsa-prefill-context-parallel \
#     --nsa-prefill-cp-mode round-robin-split \
#     --nsa-prefill-backend flashmla_auto --nsa-decode-backend flashmla_kv \
#     --kv-cache-dtype "${KV_DTYPE}" --dtype bfloat16 --mem-fraction-static 0.8 \
#     --page-size 64 --chunked-prefill-size 32768 \
#     --quantization "${QUANT}" \
#     --speculative-algorithm EAGLE --speculative-num-steps 3 \
#     --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
#     --disable-cuda-graph \
#     --disaggregation-mode prefill \
#     --disaggregation-ib-device "${IB_DEVICES}" \
#     --deepep-config "${EP_CONFIG_JSON}"
#   # ⚠ 2P1D 时，每个 P 节点还需各加一个唯一的：
#   #    --disaggregation-bootstrap-port <对应 bootstrap 端口>

# ============================================================================
# 三、Decode 节点启动（--disaggregation-mode decode）
# ============================================================================
# decode 常用 DP+EP（如 DP16+EP16），EAGLE/MTP 投机在这一侧。
#   sglang serve "${EXTRA_OPT}" \
#     --model-path "${MODEL_PATH}" --trust-remote-code \
#     --served-model-name "${MODEL_NAME}" \
#     --host "${D_IP}" --port "${D_PORT}" \
#     --dist-init-addr "${D_MASTER_IP}:${D_INIT_PORT}" \
#     --nnodes <N> --node-rank <RANK> \
#     --tp-size <TP> --dp-size <DP> --ep-size <EP> \
#     --moe-dense-tp-size 1 \
#     --enable-dp-attention --enable-dp-lm-head \
#     --moe-a2a-backend deepep --deepep-mode low_latency \
#     --context-length "${CTX_LEN}" \
#     --dtype bfloat16 --mem-fraction-static 0.8 \
#     --page-size 64 --chunked-prefill-size -1 \
#     --kv-cache-dtype "${KV_DTYPE}" --quantization "${QUANT}" \
#     --disable-radix-cache \
#     --cuda-graph-max-bs <BS> --max-running-requests <REQS> \
#     --speculative-algorithm EAGLE --speculative-num-steps 3 \
#     --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
#     --disaggregation-mode decode \
#     --disaggregation-ib-device "${IB_DEVICES}"
#   # 注：多节点（--nnodes >1）时 dist-init-addr 用主节点地址，node-rank 逐节点 0/1/...

# ============================================================================
# 四、Router（PD 分离路由）
# ============================================================================
# 1P1D（1 prefill + 1 decode），cache_aware 策略：
#   python -m sglang_router.launch_router \
#     --pd-disaggregation \
#     --prefill http://${P_IP}:${P_PORT} \
#     --decode http://${D_IP}:${D_PORT} \
#     --policy cache_aware --port ${ROUTER_PORT}

# 2P1D（2 prefill + 1 decode）——策略必须改 round_robin；
#   每个 P 节点各带一个 --disaggregation-bootstrap-port（见上）。
#   python -m sglang_router.launch_router \
#     --pd-disaggregation \
#     --prefill http://${P1_IP}:${P1_PORT} <P1_BOOT_PORT> \
#     --prefill http://${P2_IP}:${P2_PORT} <P2_BOOT_PORT> \
#     --decode http://${D_IP}:${D_PORT} \
#     --policy round_robin --port ${ROUTER_PORT}

# ============================================================================
# 五、DeepEP ep_config.json（--deepep-config 指向）
# ============================================================================
# normal_dispatch / normal_combine 可分别调 SM 数与 chunked token 数：
#   {
#     "normal_dispatch": { "num_sms": 48,
#       "num_max_nvl_chunked_send_tokens": 6,   "num_max_nvl_chunked_recv_tokens": 256,
#       "num_max_rdma_chunked_send_tokens": 6,  "num_max_rdma_chunked_recv_tokens": 128 },
#     "normal_combine":   { "num_sms": 48,
#       "num_max_nvl_chunked_send_tokens": 4,   "num_max_nvl_chunked_recv_tokens": 256,
#       "num_max_rdma_chunked_send_tokens": 6,  "num_max_rdma_chunked_recv_tokens": 128 }
#   }
# 若用 static 专家分发还需 `--ep-dispatch-algorithm static --ep-num-redundant-experts <N>
#   --eplb-algorithm deepseek_vec`，并用 expert_distribution 记录文件。

# ============================================================================
# 六、单机部署（非 PD 分离，多模态 / 单卡）
# ============================================================================
# 多模态（SGLang，fa3 attention + 多模态 feature）：
#   sglang serve "${MODEL_PATH}" --trust-remote-code \
#     --attention-backend fa3 --mm-attention-backend fa3 \
#     --keep-mm-feature-on-device --dtype bfloat16 \
#     --kv-cache-dtype "${KV_DTYPE}" --page-size 64 \
#     --enable-piecewise-cuda-graph --mem-fraction-static 0.8

# vLLM 多模态（mm-processor 缓存）：
#   vllm serve "${MODEL_PATH}" --trust-remote-code -tp <N> \
#     --kv-cache-dtype fp8_e4m3 --mm-processor-cache-type shm \
#     --mm-processor-cache-gb 8 --max-model-len <LEN> \
#     --gpu-memory-utilization 0.90 --no-enable-prefix-caching

# 容器环境（DCU 必挂：-v /opt/hyhal:/opt/hyhal:ro）：
#   docker run -dit --shm-size <SHM> --network=host --name=<NAME> \
#     --privileged --gpus all --group-add video \
#     --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
#     -v /opt/hyhal:/opt/hyhal:ro \
#     -v <DATA_DIR>:/data <REGISTRY>/<SGLANG_IMG> bash
