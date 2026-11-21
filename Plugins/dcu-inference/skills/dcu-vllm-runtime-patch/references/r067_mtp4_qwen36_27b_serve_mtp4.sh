#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${MODEL_DIR:?Set MODEL_DIR to the DCU-adapted Qwen3.6-27B-W8A8-DCU directory}"

IMAGE="${IMAGE:-<internal_registry>/dcu/admin/base/custom:vllm018-ubuntu22.04-dtk26.04-qwen3.6-20260423@sha256:13ce550647063a7fe76e87fd173986175946e5046bd36980c4289c60a4bdd811}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen36-27b-k100ai-r067-mtp4}"
CACHE_DIR="${CACHE_DIR:-$ROOT_DIR/.cache/r067-mtp4}"
MODEL_DIR="$(cd "$MODEL_DIR" && pwd)"
PATCH_DIR="$ROOT_DIR/patches/r067_mtp4"
NATIVE_SO="$ROOT_DIR/native/k100_int8_gemv_v6.so"
DCU_CONFIG_SHA256="481eaef70106fb335db108707b0fb79db0efe543d8413b6dd3fa9ebb92990724"

if [[ ! -f "$NATIVE_SO" ]]; then
  echo "ERROR: native extension not found: $NATIVE_SO" >&2
  echo "Build it first: bash $ROOT_DIR/scripts/build_native.sh" >&2
  exit 2
fi

if command -v sha256sum >/dev/null 2>&1; then
  MODEL_CONFIG_SHA256="$(sha256sum "$MODEL_DIR/config.json" | awk '{print $1}')"
else
  MODEL_CONFIG_SHA256="$(shasum -a 256 "$MODEL_DIR/config.json" | awk '{print $1}')"
fi
if [[ "$MODEL_CONFIG_SHA256" != "$DCU_CONFIG_SHA256" ]]; then
  echo "ERROR: MODEL_DIR is not using the validated 27B DCU config." >&2
  echo "Run: python3 $ROOT_DIR/scripts/apply_dcu_config.py --model-dir $MODEL_DIR" >&2
  exit 2
fi

mkdir -p "$CACHE_DIR"
devices=(--device=/dev/kfd --device=/dev/dri)
[[ -e /dev/mkfd ]] && devices+=(--device=/dev/mkfd)

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run -d \
  --name "$CONTAINER_NAME" \
  --network host \
  --ipc host \
  --shm-size 16g \
  --privileged \
  "${devices[@]}" \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES="$GPU_ID" \
  -e PYTHONPATH=/opt/k100-27b-patch \
  -e K100_27B_SC_LINEAR_A8W8=1 \
  -e K100_27B_SC_LINEAR_A8W8_MAX_M=5 \
  -e K100_27B_LM_HEAD_W8A8=1 \
  -e K100_27B_LM_HEAD_W8A8_MAX_M=5 \
  -e K100_27B_GDN_FUSED=1 \
  -e K100_27B_NATIVE_OUT_GEMV=1 \
  -e K100_27B_NATIVE_OUT_GEMV_SO=/opt/k100-27b-native/k100_int8_gemv_v6.so \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "$MODEL_DIR":/models/qwen36-27b-w8a8:ro \
  -v "$CACHE_DIR":/root/.cache/vllm \
  -v "$PATCH_DIR":/opt/k100-27b-patch:ro \
  -v "$ROOT_DIR/native":/opt/k100-27b-native:ro \
  "$IMAGE" \
  vllm serve /models/qwen36-27b-w8a8 \
    --host 0.0.0.0 \
    --port "$PORT" \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --quantization compressed-tensors \
    --served-model-name qwen36-27b-w8a8-k100ai \
    --language-model-only \
    --generation-config vllm \
    --disable-custom-all-reduce \
    -cc.mode=3 \
    -cc.inductor_compile_config '{"combo_kernels": false, "benchmark_combo_kernel": false}' \
    --cudagraph-capture-sizes 1 5 \
    --max-num-seqs 32 \
    --max-num-batched-tokens 4096 \
    --speculative-config '{"model":"/models/qwen36-27b-w8a8","method":"qwen3_next_mtp","num_speculative_tokens":4,"quantization":"compressed-tensors"}'

echo "Started $CONTAINER_NAME on GPU $GPU_ID, port $PORT (R067 + MTP4)."
echo "Accepted fixed-512 reference: ~42.30 tok/s median on the validated K100AI stack."
echo "MTP throughput depends on prompt and speculative-token acceptance."
echo "Follow startup: docker logs -f $CONTAINER_NAME"
