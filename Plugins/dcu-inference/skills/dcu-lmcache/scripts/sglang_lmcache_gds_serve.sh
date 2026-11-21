#!/usr/bin/env bash
# SGLang + LMCache GDS(hyfile/XDS) 启动与四组对照压测脚本
# 来源：内部文档《XDS × LMCache 技术文档》(file_id <internal-file-id>, 2026-08 实测)
# 实测平台：SGLang 0.5.12-ubuntu22.04-dtk2604-py3.10、Qwen3-32B (TP=1)、DCU BW1100、DTK2604
#
# 重要前置（脚本不替代，需先做）：
#   1) 宿主机装 DDS：bash ./dds-3.4.0.0.run -i，确认 lsmod | grep hyfs、/dev/hyfs 字符设备存在
#   2) 容器透传：--device=/dev/hyfs -v /opt/hyhal:/opt/hyhal（见 dcu-lmcache SKILL.md §XDS 环境要求）
#   3) 容器内 source /opt/dtk/env.sh /opt/hyhal/env.sh，LD_LIBRARY_PATH 含 /opt/hyhal/lib
#   4) LMCache 含 hyfile 适配，python -c "from lmcache.v1.storage_backend.hyfile_ctypes import has_hyfile; assert has_hyfile()"
#
# 危险：start_case 内 pkill 会停掉容器内匹配的 SGLang 进程，仅在本验证容器执行，勿用于共享业务容器。
set -u

# ---------- 环境与模型（按实际改） ----------
export MODEL_PATH=/data/models/Qwen3-32B
export PORT=10025
export TP=1
export CUDA_VISIBLE_DEVICES=0

# Serving 统一参数（文档 §3.2 / 附录 B）
export MEM_FRACTION_STATIC=0.55
export MAX_TOTAL_TOKENS=24576
export MAX_RUNNING_REQUESTS=32

# 压测负载（generated-shared-prefix 数据集，文档 §3.2 / 附录 B）
export GSP_SYS=12288     # 每组共享 system 前缀 token 数
export GSP_Q=64         # 每个请求问题 token 数
export GSP_OUT=16       # 输出 token 数（故意短，减少 decode 对 TTFT 稀释）
export GSP_GROUPS=4     # 独立共享前缀组数
export GSP_PPG=8        # 每组问题数，总请求 = groups*ppg = 32
export CONC=16          # 最大并发
export NUM_PROMPTS=32

# ---------- 启动 SGLang（按 case 切换 LMCache 配置） ----------
start_case() {
  local case_id="$1"
  local -a lmcache_args

  pkill -f 'python -m sglang.launch_server' || true
  sleep 2

  case "${case_id}" in
    1)
      # 无 LMCache 基线
      unset LMCACHE_CONFIG_FILE LMCACHE_USE_EXPERIMENTAL
      lmcache_args=(--disable-radix-cache)
      ;;
    2)
      # CPU 内存热层（configs 见 dcu-lmcache lmcache_config_cpu.yaml 思路）
      export LMCACHE_USE_EXPERIMENTAL=True
      export LMCACHE_CONFIG_FILE=configs/case2_lmcache_cpu.yaml
      lmcache_args=(--enable-lmcache)
      ;;
    3)
      # NVMe POSIX
      export LMCACHE_USE_EXPERIMENTAL=True
      export LMCACHE_CONFIG_FILE=configs/case3_lmcache_nvme.yaml
      find /data/nvme-lmcache -mindepth 1 -delete
      lmcache_args=(--enable-lmcache)
      ;;
    4)
      # GDS hyfile（见 scripts/lmcache_config_gds_hyfile.yaml）
      export LMCACHE_USE_EXPERIMENTAL=True
      export LMCACHE_CONFIG_FILE=configs/case4_lmcache_gds.yaml
      find /data/gds-lmcache -mindepth 1 -delete
      lmcache_args=(--enable-lmcache)
      ;;
    *)
      echo "unknown case: ${case_id}" >&2
      return 2
      ;;
  esac

  nohup python -m sglang.launch_server \
    --trust-remote-code \
    --model-path "${MODEL_PATH}" \
    --tp "${TP}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --chunked-prefill-size 8192 \
    --cuda-graph-max-bs 32 \
    --page-size 64 \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    --max-total-tokens "${MAX_TOTAL_TOKENS}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --skip-server-warmup \
    "${lmcache_args[@]}" \
    >"bench2/case${case_id}_server.log" 2>&1 &

  for i in $(seq 1 120); do
    if curl -sf http://127.0.0.1:${PORT}/v1/models >/dev/null; then
      echo "READY case${case_id}"
      return 0
    fi
    sleep 5
  done

  echo "case${case_id} did not become ready" >&2
  return 1
}

# ---------- 单轮压测（write=首轮写缓存；hit=清 GPU radix 后命中回填） ----------
run_round() {
  local case_id="$1"
  local round="$2"
  local warmup output_suffix
  local -a initial_flush

  case "${round}" in
    write)
      warmup=1
      output_suffix="gsp"
      initial_flush=(--flush-cache)
      ;;
    hit)
      warmup=0
      output_suffix="gsp_hit"
      initial_flush=()
      ;;
    *)
      echo "unknown round: ${round}" >&2
      return 2
      ;;
  esac

  python -m sglang.bench_serving \
    --backend sglang \
    --base-url http://127.0.0.1:${PORT} \
    --model "${MODEL_PATH}" \
    --dataset-name generated-shared-prefix \
    --gsp-num-groups "${GSP_GROUPS}" \
    --gsp-prompts-per-group "${GSP_PPG}" \
    --gsp-system-prompt-len "${GSP_SYS}" \
    --gsp-question-len "${GSP_Q}" \
    --gsp-output-len "${GSP_OUT}" \
    --num-prompts "${NUM_PROMPTS}" \
    --max-concurrency "${CONC}" \
    --warmup-requests "${warmup}" \
    "${initial_flush[@]}" \
    --output-file "bench2/case${case_id}_${output_suffix}.json" \
    2>&1 | tee "bench2/case${case_id}_${output_suffix}.log"
}

# ---------- 主流程：CASE_ID 默认 4（GDS），可传入 1..4 ----------
CASE_ID="${1:-4}"

start_case "${CASE_ID}"
run_round "${CASE_ID}" write

# 只清 GPU radix/KV；当前版本下 LMCache 文件应保留（flush 语义随版本变，需实测确认）
curl -sf -X POST http://127.0.0.1:${PORT}/flush_cache
sleep 2

run_round "${CASE_ID}" hit

grep -E 'Mean TTFT|Median TTFT|Request throughput|Input token throughput|Mean E2E' \
  "bench2/case${CASE_ID}_gsp_hit.log"

pkill -f 'python -m sglang.launch_server' || true
