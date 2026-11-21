# E=256,N=512,device_name=K100_AI.json — MoE 调优配置（K100 AI / Qwen3.6-35B-A3B W8A8）

## 来源
本地复现工程 <local_repro_pkg>_2026-08-07.gz
  config/E=256,N=512,device_name=K100_AI.json
本文件是 vLLM fused_moe 的 **per-M 形状调优表**，逐字节原样提取，保持合法 JSON，
可被 vLLM 直接挂载（不夹注释，以免影响解析）。

## 安全说明
纯配置，无密码 / 密钥 / 凭据。

## 内容含义
顶层 key 为 decode 的 M 档（1~5），value 为对应 fused MoE GEMM 的 Triton/AITER
BLOCK_SIZE_* / warp / stage 参数：
  - "1".."5": 各 M 档的 MoE tile 配置（BLOCK_SIZE_M/N/K、GROUP_SIZE_M、SPLIT_K、
    num_warps、num_stages、waves_per_eu、matrix_instr_nonkdim、kpack）
Qwen3.6 的 MoE 形状：256 专家 / 激活 8 / E=512（intermediate=512）/ N=2048（down 输出）。

## 使用前提（离开 K100 AI / gfx928 / vLLM 0.18.1 / DTK 26.04 须重新验证）
- 挂载到 vLLM 的 fused_moe configs 目录：
    -v $THIS_JSON:/usr/local/lib/python3.10/dist-packages/vllm/model_executor/layers/fused_moe/configs/E=256,N=512,device_name=K100_AI.json:ro
  （容器 Python 固定 3.10；路径以实际镜像为准）
- 该配置需配合 native INT8 MoE kernel（references/k100_channelwise_int8_moe.hip +
  build_k100_channelwise_int8_moe.py）及 K100_NATIVE_INT8_MOE=1 启用。
- AITER 模板必须 patch group_size_n==1 为真 per-output-channel（否则数值错，详见
  dcu-perf-cases Case 10 与 SKILL.md §4 说明）。
- 文件名中的 K100_AI 是 device_name 标识，vLLM 按机型名匹配该配置；换机型勿直接套用。

## 关联文件（同 references/）
- k100_channelwise_int8_moe.hip
- build_k100_channelwise_int8_moe.py
- r180_nomtp_sitecustomize.py / r184_mtp3_fused_qkvz_ba_sitecustomize.py
- start_r180_nomtp_gpu1_speedtest.sh / start_final_r184_mtp3_gpu0_speedtest.sh
