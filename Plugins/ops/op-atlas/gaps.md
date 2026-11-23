# GAPS —— 已知未覆盖与证据缺口（生成物，勿手改）

> 由 `scripts/gen_gaps.py` 生成。改这张表请改脚本或卡片，不要直接编辑本文件。
> 生成时间：2026-11-23 12:22
>
> **它防的是「用覆盖数掩盖深度不足」。** 46 张卡容易被当成全覆盖，
> 实际有 13 张挂不上可复跑证据、5 个家族零实测。

## 一、证据缺口（`missing_evidence`）

`measured` 的本义是「有数**且可追**」。下列 13 张卡有结论但挂不上可复跑的 results 文件（E25 判 missing）。

| 卡片 | 连续缺证据批数 | 处置 |
|---|---|---|
| `flash-attention-fwd` | 2 | **降级 draft**（连续 ≥2 批） |
| `fused-add-rmsnorm` | 2 | **降级 draft**（连续 ≥2 批） |
| `gemm-int8` | 2 | **降级 draft**（连续 ≥2 批） |
| `layernorm` | 2 | **降级 draft**（连续 ≥2 批） |
| `mlp-gelu-geglu` | 2 | **降级 draft**（连续 ≥2 批） |
| `moe-combine-reduce` | 2 | **降级 draft**（连续 ≥2 批） |
| `moe-swiglu-fused` | 2 | **降级 draft**（连续 ≥2 批） |
| `moe-w8a8` | 2 | **降级 draft**（连续 ≥2 批） |
| `mqa-logits` | 2 | **降级 draft**（连续 ≥2 批） |
| `qk-norm` | 2 | **降级 draft**（连续 ≥2 批） |
| `rope` | 2 | **降级 draft**（连续 ≥2 批） |
| `rope-mla` | 2 | **降级 draft**（连续 ≥2 批） |
| `sparse-mla` | 2 | **降级 draft**（连续 ≥2 批） |

## 二、零实测家族

| 家族 | 卡数 | 为什么零实测 |
|---|---|---|
| `kv` | 3 | 3 张卡全 draft；paged 写入的测量被 launch 开销淹没（见 case 009 的同类问题） |
| `linear-attn` | 6 | 9 条映射已有，但 0 张实测；本机未找到可跑通的 chunked-scan / gated-linear-attn 入口 |
| `memory` | 1 | embedding-lookup 1 张 draft；属访存类，判据与 rope 同源但未跑 |
| `quant` | 2 | 4 条映射 0 实测；per-token/per-block 判据未落地 |
| `sampling` | 3 | 3 张卡全 draft；判据未定（top-k 与 top-p 的截断语义不同） |

有实测的家族 7 / 12：`activation`、`attention`、`elementwise`、`gemm`、`moe`、`norm`、`rope`

## 二b、分母未标定

下列常数被当作达成率的分母，但**全仓没有可复跑的测量记录**。
在补上标定前，凡用它们算出的 utilization 必须与实测峰值并列双报。

| 常数 | 值 | 现状 | 实测参考 | 补的前提 |
|---|---|---|---|---|
| `memory.stream_read_peak` | 1724 GB/s | **只有断言，无测量记录** | `scripts/bw_calibrate.py` 实测：copy 1051 / mul 1287 / add 1634 / sum 865 GB/s（2026-11-23，sclk 1350），**无一到 1724** | 跑 `bw_calibrate.py` 并据实改值，或标 unverified |

> 这是「分母错了 10 倍」那类问题的第三次：1100 标称 → 185 反推 → 540 无来源。
> 前三次都是「分母的来源不是本仓库能跑的测量」。现在补了 `bw_calibrate.py`，
> 但 1724 这个数本身还没被替换——它需要先跑一次并确认时钟档位。

## 三、已知未覆盖算子

| 未覆盖 | 为什么没覆盖 | 补的前提 |
|---|---|---|
| `all_reduce / all_gather` | 属 RCCL，本仓库改不动其 kernel | 只能做重叠编排（归 engine 层，不是算子卡） |
| `dequant_fp8_w4` | 无公开入口，本机也没有可调用的 w4a16 反量化算子 | 需先有 w4a16 量化卡与可跑通的权重格式 |
| `sampling_top_p` | 映射有、实测无；top-p 的判据（累积概率截断点）与 top-k 不同 | 需先定 sampling 族的判据，再建 case |
| `moe_grouped_fp8_ll` | deepgemm 的 ll 路径要求 6D packed layout，本机造不出 | 需先解决 pack 格式（case 002 记为未测定） |
| `flash_mla sparse prefill 形态` | 本机 flash_mla 只跑通 decode 剂量 | 需确认真实 trace 里 prefill 是否真走这条路径 |

## 四、当前实测覆盖

有数卡 24 / 全部 46 张。其中证据齐全 11 张、缺证据 13 张。

