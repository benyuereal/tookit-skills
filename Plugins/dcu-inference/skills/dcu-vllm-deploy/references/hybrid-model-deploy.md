# Hybrid 线性注意力模型部署参考（qwen3_5/qwen3_next 系）

> 来源：VeriLoop-E2（清华 SIGS，Qwen3_5ForConditionalGeneration）海光 DCU vLLM 0.28 实测部署，2026-10。公开可复现：公开镜像 hcu/admin/base/custom:vllm0.28.1rc1-ubuntu22.04-dtk2604-py3.10-20260912-1007 + ModelScope 权重。

## 架构识别

- config.json `architectures: [Qwen3_5ForConditionalGeneration]`：64 层 = 48 层 GDN 线性注意力 + 每 4 层插 1 层全注意力 + 1 层 MTP。
- GDN 头参数：linear_num_key_heads=16 / linear_num_value_heads=48 / head_dim=128 / conv_kernel=4；全注意力 heads 24 / KV 4 / hidden 5120。
- vLLM 侧 `Qwen3_5ForConditionalGeneration` 原生注册（0.28.1rc1 探针确认，含 Moe/MTP 变体）。
- 仓库常带 VLM 壳（video_preprocessor_config.json）——纯文本推理加 `--language-model-only`。

## TP 合法性通式

全 attention heads 集合 % TP == 0 且 KV heads % TP == 0（或 TP % KV == 0）；GDN 额外要求 linear_num_key_heads / linear_num_value_heads % TP == 0。

例：16/48/24/4/5120 → TP ∈ {1,2,4,8}。

## 启动命令（已验证，公开镜像）

```bash
docker run -d --name veriloop-e2 \
  --device=/dev/kfd --device=/dev/mkfd --device=/dev/dri \
  --group-add video --shm-size=16g \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v <模型目录>:/model:ro \
  -e HIP_VISIBLE_DEVICES=<卡列表> \
  -e VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0 \
  -p 8000:8000 \
  harbor.sourcefind.cn:5443/hcu/admin/base/custom:vllm0.28.1rc1-ubuntu22.04-dtk2604-py3.10-20260912-1007 \
  bash -lc "source /opt/dtk/env.sh && vllm serve /model \
    --tensor-parallel-size 4 --dtype bfloat16 --max-model-len 131072 \
    --max-num-batched-tokens 8192 --gpu-memory-utilization 0.90 \
    --language-model-only --port 8000"
```

三坑说明见 SKILL.md §4.5（retention 整除性 / VLM 壳 / libcudart shim 按镜像而定）。

## 精度验收

门禁方法（HF 基线 + token 级 diff + 四组判别实验 + 分级放行）见 `dcu/troubleshooting/cross-engine-acceptance-gate`。

## 版本观测值（仅注记，非通用基线）

- 内部 0.28.0-r3：eager 19/23、graph 15/23（vs HF bf16 基线，分叉全语义等价）
- 公开 0.28.1rc1：eager 17/23（新增分叉在长文 echo 边界，语义零差）
- graph 比 eager 多分叉是 cudagraph 融合 kernel 数值差，预期行为
