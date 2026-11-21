# =============================================================================
# 实战启动脚本（K100 AI / gfx928 / vLLM 0.18.1 / DTK 26.04 / Qwen3.6-35B-A3B W8A8）
# -----------------------------------------------------------------------------
# 来源：本地复现工程 <local_repro_pkg>_2026-08-07.gz
#        - scripts/start_r180_nomtp_gpu1_speedtest.sh   (无 MTP 纯模型 R180 启动, GPU1/端口7995)
#        - scripts/start_final_r184_mtp3_gpu0_speedtest.sh  (MTP3 版 R184 启动, GPU0/端口7994)
# 安全说明：本文件为真实部署脚本，不含有任何密码 / 密钥 / 凭据。
#           IP / 端口 / 镜像 / 模型路径 / 缓存路径均为项目公开信息，无内网凭证。
#           docker run 已挂 --device=/dev/kfd --device=/dev/dri -v /opt/hyhal:/opt/hyhal:ro
#           （DCU 容器必挂 hyhal，否则 DEVCOUNT=0）。
# 使用前提（离开以下环境须相应调整）：
#   - 机型：海光 K100 AI（gfx928）；启动用 HIP_VISIBLE_DEVICES 指定单卡
#   - 镜像：<internal_registry>/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-qwen3.6-20260423
#   - 模型：/data/my_model/Qwen3.6-35B-A3B-W8A8-DCU（compressed-tensors W8A8）
#   - 补丁：PYTHONPATH 指向 runtime_patch 与 native_ext（见本 skill references/ 的两个
#           sitecustomize 及配套 native_ext/.hip；MoE 调优 json 挂到 vLLM fused_moe configs）
# 配套 artifact（本 skill 不含，需从复现工程取得）：
#   - runtime_patch_r180_nomtp/ 或 patches/runtime_patch_r184_qkvz_ba_fused/（含 sitecustomize）
#   - native_ext/k100_channelwise_int8_moe.hip + build_k100_channelwise_int8_moe.py
#   - config/E=256,N=512,device_name=K100_AI.json
# 注意：脚本内 GPU/端口/容器名是为复现测试隔离设定的，照搬前按需改；生产服务勿误停。
# =============================================================================

#!/usr/bin/env bash
set -euo pipefail

NAME=qwen36-final-mtp3-r184-speedtest
CACHE=/data/qwen36-k100-int8-opt/cache/r184-mtp3-fused-qkvz-ba
PATCH=/data/qwen36-k100-int8-opt/runtime_patch_r184_qkvz_ba_fused
MOE_CONFIG=/data/qwen36-k100-int8-opt/experiments/R176_mtp3_m1_tuned/E=256,N=512,device_name=K100_AI.json
IMAGE=<internal_registry>/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-qwen3.6-20260423
MODEL=/data/my_model/Qwen3.6-35B-A3B-W8A8-DCU
PORT=7994
GPU=0

# Final isolated speed-test service. Its name is intentionally outside the
# experimental r18x/r19x cleanup patterns so parallel candidate scripts cannot
# remove it while llm_speedtest is running.
docker rm -f qwen36-w8a8-sc-mtp3-r184 qwen36-w8a8-sc-mtp3-r189 qwen36-w8a8-sc-mtp3-r191 qwen36-w8a8-sc-mtp3-r192 qwen36-w8a8-sc-mtp3-r193 >/dev/null 2>&1 || true
docker rm -f "$NAME" >/dev/null 2>&1 || true
mkdir -p "$CACHE"

docker run -d \
  --name "$NAME" \
  --network host \
  --ipc host \
  --shm-size 16g \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  -e HIP_VISIBLE_DEVICES="$GPU" \
  -e TZ=Asia/Shanghai \
  -e PYTHONPATH=/opt/k100-runtime-patch:/opt/k100-native-ext \
  -e K100_NATIVE_INT8_MOE=0 \
  -e K100_NATIVE_INT8_MOE_MAX_M=32 \
  -e K100_SC_LINEAR_A8W8=1 \
  -e K100_SC_LINEAR_A8W8_MAX_M=4 \
  -e K100_LM_HEAD_W8A8=1 \
  -e K100_LM_HEAD_W8A8_MAX_M=4 \
  -e K100_FUSED_QKVZ_BA=1 \
  -e K100_LOG_LINEAR_LAYOUT=0 \
  -e K100_TRACE_LINEAR_SHAPES=0 \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v /data:/data \
  -v "$CACHE":/root/.cache/vllm \
  -v "$PATCH":/opt/k100-runtime-patch:ro \
  -v /data/qwen36-k100-int8-opt/native_ext:/opt/k100-native-ext:ro \
  -v "$MOE_CONFIG":/usr/local/lib/python3.10/dist-packages/vllm/model_executor/layers/fused_moe/configs/E=256,N=512,device_name=K100_AI.json:ro \
  "$IMAGE" \
  vllm serve "$MODEL" \
    --port "$PORT" \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --quantization compressed-tensors \
    --served-model-name qwen36-35b-a3b-w8a8-native-k100 \
    --language-model-only \
    --generation-config vllm \
    --disable-custom-all-reduce \
    -cc.mode=3 \
    -cc.inductor_compile_config '{"combo_kernels": false, "benchmark_combo_kernel": false}' \
    --cudagraph-capture-sizes 1 4 \
    --max-num-seqs 32 \
    --speculative-config '{"model":"/data/my_model/Qwen3.6-35B-A3B-W8A8-DCU","method":"qwen3_next_mtp","num_speculative_tokens":3,"quantization":"compressed-tensors"}'
