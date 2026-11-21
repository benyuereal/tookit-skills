#!/bin/bash
# TileLang RMSNorm 端到端 profiling 流水线（源自 内部文档 主文档 §3.1）
# 用途: 起 vLLM 服务 → 压测 → 用 hipprof --hip-trace 包裹整个流程，
#       产出 Chrome Trace / .hipkernel.csv / .hiptrace.csv 用于算子耗时分析。
# 平台: DCU 海光; 须在挂载 /opt/hyhal 的 DCU 容器内运行（否则 DCU 不可见）。
# 溯源: 内部文档 主文档《Tilelang算子开发实践》file_id=<internal-file-id> §3.1
#
# ⚠ 路径占位（按本机环境替换）:
#   MODEL_BASE  -> 模型根目录（文档原值 /data1/models）
#   DATASET_PATH-> ShareGPT 数据集（文档原值 /data1/wyt/quantize/benchmark/datasets/ShareGPT_V3_unfiltered_cleaned_split.json）
#   HIP_VISIBLE_DEVICES -> 指定用哪几张卡（编号从 0 起；DCU 纪律用此变量挑卡）
#
# 产物:
#   result_*.json          Chrome Trace Event 格式 HIP Runtime API 时间线
#   result_*.hipkernel.csv GPU Kernel 统计（调用次数/总耗时/平均/占比）
#   result_*.hiptrace.csv  HIP API 调用统计
# 观察: 原生 RMSNorm 在 PyTorch 分解为多个 element-wise/reduce kernel，
#       不会以单一算子名出现；Triton 融合版名字含 mean/mul/pow/rsqrt 即 RMSNorm。

set -u

MODEL_BASE="${MODEL_BASE:-/data1/models}"
RESULTS_DIR="${RESULTS_DIR:-./experiment_results}"
DATASET_PATH="${DATASET_PATH:-/data1/wyt/quantize/benchmark/datasets/ShareGPT_V3_unfiltered_cleaned_split.json}"
CONCURRENCY=10
NUM_REQUESTS=100

# 指定卡（DCU 纪律: 用 HIP_VISIBLE_DEVICES 挑卡，不靠 -g 自挑）
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"

EXPERIMENTS=(
    "Qwen3-8B|none|FP16_Baseline"
)

mkdir -p "$RESULTS_DIR"

# hipprof 包裹整段: 服务启动 + 压测 都在 hip-trace 跟踪范围内
hipprof --hip-trace --output-type 0 bash -c '
MODEL_BASE="'"$MODEL_BASE"'"
RESULTS_DIR="'"$RESULTS_DIR"'"
DATASET_PATH="'"$DATASET_PATH"'"
CONCURRENCY='"$CONCURRENCY"'
NUM_REQUESTS='"$NUM_REQUESTS"'
EXPERIMENTS=("'"${EXPERIMENTS[*]}"'")

for exp in "${EXPERIMENTS[@]}"; do
    IFS="|" read -r SUBDIR QUANT TAG <<< "$exp"
    MODEL_PATH="$MODEL_BASE/$SUBDIR"
    LOG_FILE="$RESULTS_DIR/${TAG}_bench.log"

    echo "========================================================="
    echo "🚀 启动实验: $TAG"

    START_TIME=$SECONDS

    vllm serve "$MODEL_PATH" \
        --port 8000 \
        --no-enable-chunked-prefill \
        --no-enable-prefix-caching \
        --dtype float16 \
        --trust-remote-code > "$RESULTS_DIR/${TAG}_server.log" 2>&1 &
    VLLM_PID=$!

    echo "⏳ 等待服务加载模型..."
    while ! curl -s http://localhost:8000/health > /dev/null; do
        sleep 1
    done

    END_TIME=$SECONDS
    LOAD_TIME=$((END_TIME - START_TIME))
    echo "⏱️ 模型加载完成！耗时: ${LOAD_TIME} 秒"
    echo "Model Loading Time: ${LOAD_TIME} s" >> "$LOG_FILE"

    echo "📊 运行 Benchmark..."
    vllm bench serve \
        --backend vllm \
        --model "$MODEL_PATH" \
        --dataset-name sharegpt \
        --dataset-path "$DATASET_PATH" \
        --request-rate "$CONCURRENCY" \
        --num-prompts "$NUM_REQUESTS" >> "$LOG_FILE" 2>&1

    kill -15 $VLLM_PID
    sleep 10
done
'
