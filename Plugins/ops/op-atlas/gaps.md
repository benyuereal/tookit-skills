# GAPS —— 已知未覆盖与证据缺口（生成物，勿手改）

> 由 `scripts/gen_gaps.py` 生成。改这张表请改脚本或卡片，不要直接编辑本文件。
> 生成时间：2026-11-23 11:38
>
> **它防的是「用覆盖数掩盖深度不足」。** 46 张卡容易被当成全覆盖，
> 实际有 14 张挂不上可复跑证据、5 个家族零实测。

## 一、证据缺口（`missing_evidence`）

`measured` 的本义是「有数**且可追**」。下列 14 张卡有结论但挂不上可复跑的 results 文件（E25 判 missing）。

| 卡片 | 连续缺证据批数 | 处置 |
|---|---|---|
| `flash-attention-fwd` | 1 | 本批 warning，下批仍缺则降级 |
| `fused-add-rmsnorm` | 1 | 本批 warning，下批仍缺则降级 |
| `gemm-int8` | 1 | 本批 warning，下批仍缺则降级 |
| `layernorm` | 1 | 本批 warning，下批仍缺则降级 |
| `mlp-gelu-geglu` | 1 | 本批 warning，下批仍缺则降级 |
| `moe-combine-reduce` | 1 | 本批 warning，下批仍缺则降级 |
| `moe-swiglu-fused` | 1 | 本批 warning，下批仍缺则降级 |
| `moe-w8a8` | 1 | 本批 warning，下批仍缺则降级 |
| `mqa-logits` | 1 | 本批 warning，下批仍缺则降级 |
| `qk-norm` | 1 | 本批 warning，下批仍缺则降级 |
| `rmsnorm-fused` | 1 | 本批 warning，下批仍缺则降级 |
| `rope` | 1 | 本批 warning，下批仍缺则降级 |
| `rope-mla` | 1 | 本批 warning，下批仍缺则降级 |
| `sparse-mla` | 1 | 本批 warning，下批仍缺则降级 |

## 二、零实测家族

| 家族 | 卡数 | 为什么零实测 |
|---|---|---|
| `kv` | 3 | 3 张卡全 draft；paged 写入的测量被 launch 开销淹没（见 case 009 的同类问题） |
| `linear-attn` | 6 | 9 条映射已有，但 0 张实测；本机未找到可跑通的 chunked-scan / gated-linear-attn 入口 |
| `memory` | 1 | embedding-lookup 1 张 draft；属访存类，判据与 rope 同源但未跑 |
| `quant` | 2 | 4 条映射 0 实测；per-token/per-block 判据未落地 |
| `sampling` | 3 | 3 张卡全 draft；判据未定（top-k 与 top-p 的截断语义不同） |

有实测的家族 7 / 12：`activation`、`attention`、`elementwise`、`gemm`、`moe`、`norm`、`rope`

## 三、已知未覆盖算子

| 未覆盖 | 为什么没覆盖 | 补的前提 |
|---|---|---|
| `all_reduce / all_gather` | 属 RCCL，本仓库改不动其 kernel | 只能做重叠编排（归 engine 层，不是算子卡） |
| `dequant_fp8_w4` | 无公开入口，本机也没有可调用的 w4a16 反量化算子 | 需先有 w4a16 量化卡与可跑通的权重格式 |
| `sampling_top_p` | 映射有、实测无；top-p 的判据（累积概率截断点）与 top-k 不同 | 需先定 sampling 族的判据，再建 case |
| `moe_grouped_fp8_ll` | deepgemm 的 ll 路径要求 6D packed layout，本机造不出 | 需先解决 pack 格式（case 002 记为未测定） |
| `flash_mla sparse prefill 形态` | 本机 flash_mla 只跑通 decode 剂量 | 需确认真实 trace 里 prefill 是否真走这条路径 |

## 四、当前实测覆盖

有数卡 24 / 全部 46 张。其中证据齐全 10 张、缺证据 14 张。

